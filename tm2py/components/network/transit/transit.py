"""Performs transit assignment and generates transit skims.

"""

from __future__ import annotations

from collections import defaultdict as _defaultdict, OrderedDict
from copy import deepcopy as _copy
from contextlib import contextmanager as _context
import json as _json
import numpy as np
import pandas as pd
import os
import copy
from typing import TYPE_CHECKING, Dict, List, Union

from tm2py.components.component import Component
import tm2py.emme as _emme_tools
from tm2py.emme.matrix import OMXManager
from tm2py.logger import LogStartEnd

if TYPE_CHECKING:
    from tm2py.controller import RunController

# TODO: imports from skim_transit_network.py, to be reviewed
import inro.modeller as _m
import inro.emme.desktop.worksheet as _worksheet


# TODO: should express these in the config
# TODO: or make global lists tuples
_all_access_modes = ["WLK_TRN_WLK", "PNR_TRN_WLK", "WLK_TRN_PNR","KNR_TRN_WLK","WLK_TRN_KNR"]


_skim_names = [
    "IWAIT",
    "XWAIT",
    "WAIT",
    "FARE",
    "BOARDS",
    "WAUX",
    "DTIME", 
    "DDIST",
    "WACC",
    "WEGR",
    "IVT",
    "IVTLOC",
    "IVTEXP",
    "IVTLRT",
    "IVTHVY",
    "IVTCOM",
    "IVTFRY",
    "CROWD",
]

_segment_cost_function = """
min_seat_weight = 1.0
max_seat_weight = 1.4
power_seat_weight = 2.2
min_stand_weight = 1.4
max_stand_weight = 1.6
power_stand_weight = 3.4

def calc_segment_cost(transit_volume, capacity, segment):
    if transit_volume <= 0:
        return 0.0
    line = segment.line
    mode_char = line{1}
    if mode_char == "p":
        congestion = 0.15 * ((transit_volume / capacity) ** 4)
    else:
        # need assignment period in seated_capacity calc?
        seated_capacity = line.vehicle.seated_capacity * {0} * 60 / line.headway
        num_seated = min(transit_volume, seated_capacity)
        num_standing = max(transit_volume - seated_capacity, 0)

        vcr = transit_volume / capacity
        crowded_factor = (((
            (min_seat_weight+(max_seat_weight-min_seat_weight)*(vcr)**power_seat_weight)*num_seated
            +(min_stand_weight+(max_stand_weight-min_stand_weight)*(vcr)**power_stand_weight)*num_standing
            )/(transit_volume)))
        congestion = max(crowded_factor, 1.0) - 1.0

    # Toronto implementation limited factor between 1.0 and 10.0
    return congestion
"""

_headway_cost_function = """
max_hdwy_growth = 1.5
max_hdwy = 999.98


def calc_eawt(segment, vcr, headway):
    # EAWT_AM = 0. 259625 + 1. 612019*(1/Headway) + 0.005274*(Arriving V/C) + 0. 591765*(Total Offs Share)
    # EAWT_MD = 0. 24223 + 3.40621* (1/Headway) + 0.02709*(Arriving V/C) + 0. 82747 *(Total Offs Share)
    line = segment.line
    prev_segment = line.segment(segment.number - 1)
    alightings = 0
    total_offs = 0
    all_segs = iter(line.segments(True))
    prev_seg = next(all_segs)
    for seg in all_segs:
        total_offs += prev_seg.transit_volume - seg.transit_volume + seg.transit_boardings
        if seg == segment:
            alightings = total_offs
        prev_seg = seg
    if total_offs < 0.001:
        total_offs = 9999  # added due to divide by zero error
    if headway < .01:
        headway = 9999
    eawt = 0.259625 + 1.612019*(1/headway) + 0.005274*(vcr) + 0.591765*(alightings / total_offs)
    # if mode is LRT / BRT mult eawt * 0.4, if HRT /commuter mult by 0.2
    # use either .mode.id or ["#src_mode"] if fares are used
    mode_char = line{0}
    if mode_char in ["l", "x"]:
        eawt_factor = 0.4
    elif mode_char in ["h", "c", "f"]:
        eawt_factor = 0.2
    else:
        eawt_factor = 1
    return eawt * eawt_factor


def calc_adj_headway(transit_volume, transit_boardings, headway, capacity, segment):
    prev_hdwy = segment["@phdwy"]
    delta_cap = max(capacity - transit_volume + transit_boardings, 0)
    adj_hdwy = min(max_hdwy, prev_hdwy * min((transit_boardings+1) / (delta_cap+1), 1.5))
    adj_hdwy = max(headway, adj_hdwy)
    return adj_hdwy

def calc_headway(transit_volume, transit_boardings, headway, capacity, segment):
    vcr = transit_volume / capacity
    eawt = calc_eawt(segment, vcr, segment.line.headway)
    adj_hdwy = calc_adj_headway(transit_volume, transit_boardings, headway, capacity, segment)
    return adj_hdwy + eawt

"""


class TransitAssignment(Component):
    """Run transit assignment and skims."""

    def __init__(self, controller: RunController):
        """Run transit assignment and skims.

        Args:
            controller: parent Controller object
        """
        super().__init__(controller)
        self._emme_manager = None
        self._num_processors = self.controller.num_processors
    
    def validate_inputs(self):
        """Validate the inputs."""
        # TODO

    @LogStartEnd("transit assignment and skims")
    def run(self):
        """Run transit assignment and skims."""
        project_path = self.get_abs_path(self.controller.config.emme.project_path)
        emme_app = self.controller.emme_manager.project(project_path)
        if not os.path.isabs(self.controller.config.emme.transit_database_path):
            emmebank_path = self.get_abs_path(self.controller.config.emme.transit_database_path)
        emmebank = self.controller.emme_manager.emmebank(emmebank_path)
        ref_scenario = emmebank.scenario(self.controller.config.time_periods[0].emme_scenario_id)
        period_names = [time.name for time in self.controller.config.time_periods]
        self.initialize_skim_matrices(period_names, ref_scenario)
        # Run assignment and skims for all specified periods
        for period in self.controller.config.time_periods:
            scenario = emmebank.scenario(period.emme_scenario_id)
            with self._setup(scenario, period):
                if self.controller.iteration >= 1:
                    self.import_demand_matrices(period.name, scenario)
                    use_ccr = self.controller.config.transit.use_ccr
                    congested_transit_assignment = self.controller.config.transit.congested_transit_assignment
                else:
                    self.create_empty_demand_matrices(period.name, scenario)
                    use_ccr = False
                    congested_transit_assignment = False
                
                use_fares = self.controller.config.transit.use_fares
                use_peaking_factor = self.controller.config.transit.use_peaking_factor           
                
                # update network before assignment
                network = scenario.get_network()

                self.update_auto_times(network, period)
                if self.controller.config.transit.get("override_connector_times", False):
                    self.update_connector_times(scenario, network, period)
                # TODO: could set attribute_values instead of full publish

                self.update_pnr_penalty(network, deflator=self.controller.config.transit.fare_2015_to_2000_deflator)

                # peaking factor
                if use_peaking_factor:
                    path_boardings = self.get_abs_path(self.controller.config.transit.output_transit_boardings_path)
                    ea_df_path = path_boardings.format(period='ea_pnr')
                    
                    if (period.name == 'am') and (os.path.isfile(ea_df_path)==False):
                        raise Exception("run ea period first to account for the am peaking factor")

                    if (period.name == 'am') and (os.path.isfile(ea_df_path)==True):
                        ea_df = pd.read_csv(ea_df_path)
                        if scenario.extra_attribute("@orig_hdw") is None:
                            scenario.create_extra_attribute("TRANSIT_LINE", "@orig_hdw")
                        if "@orig_hdw" in network.attributes("TRANSIT_LINE"):
                            network.delete_attribute("TRANSIT_LINE", "@orig_hdw")
                        network.create_attribute("TRANSIT_LINE", "@orig_hdw")

                        for line in network.transit_lines():
                            line["@orig_hdw"] = line.headway
                            line_name = line.id
                            line_veh = line.vehicle
                            line_hdw = line.headway
                            assignment_period = period.length_hours

                            line_cap = 60 * assignment_period * line_veh.total_capacity / line_hdw

                            if line_name in ea_df['line_name_am'].to_list():
                                ea_boardings = ea_df.loc[ea_df['line_name_am'] == line_name,'boardings'].values[0]
                            else:
                                ea_boardings = 0

                            pnr_peaking_factor =(line_cap-ea_boardings)/line_cap #substract ea boardings from am parking capacity
                            non_pnr_peaking_factor = self.controller.config.transit.am_peaking_factor

                            # in Emme transit assignment, the capacity is computed for each transit line as: 60 * assignment_period * vehicle.total_capacity / line.headway
                            # so instead of applying peaking factor to calculated capacity, we can divide line.headway by this peaking factor
                            # if ea number of parkers exceed the am parking capacity, set the headway to a very large number
                            if pnr_peaking_factor>0:
                                pnr_line_hdw = line_hdw/pnr_peaking_factor 
                            else:
                                pnr_line_hdw = 999  # 999  fix it later

                            non_pnr_line_hdw = line_hdw/non_pnr_peaking_factor

                            if 'pnr' in line_name and 'egr' in line_name:
                                continue
                            elif 'pnr' in line_name and 'acc' in line_name:
                                line.headway = pnr_line_hdw
                            else:
                                line.headway = non_pnr_line_hdw

                    if (period.name == 'pm'):
                        if scenario.extra_attribute("@orig_hdw") is None:
                            scenario.create_extra_attribute("TRANSIT_LINE", "@orig_hdw")
                        if "@orig_hdw" in network.attributes("TRANSIT_LINE"):
                            network.delete_attribute("TRANSIT_LINE", "@orig_hdw")
                        network.create_attribute("TRANSIT_LINE", "@orig_hdw")

                        for line in network.transit_lines():
                            line["@orig_hdw"] = line.headway
                            line_name = line.id
                            line_hdw = line.headway

                            non_pnr_peaking_factor = self.controller.config.transit.pm_peaking_factor
                            non_pnr_line_hdw = line_hdw/non_pnr_peaking_factor

                            if 'pnr' in line_name:
                                continue
                            else:
                                line.headway = non_pnr_line_hdw

                scenario.publish_network(network)

                self.assign_and_skim(
                    scenario,
                    network,
                    period=period,
                    assignment_only=False,
                    use_fares=use_fares,
                    use_ccr=use_ccr,
                    congested_transit_assignment=congested_transit_assignment
                )
                self.export_skims(period.name, scenario)

                if (period.name == 'ea') and use_peaking_factor:
                    line_name=[]
                    boards=[]
                    ea_df = pd.DataFrame()
                    network = scenario.get_network()

                    for line in network.transit_lines():
                        boardings = 0
                        for segment in line.segments(include_hidden=True):
                            boardings += segment.transit_boardings  
                        line_name.append(line.id)
                        boards.append(boardings)
                
                    ea_df["line_name"]  = line_name
                    ea_df["boardings"]  = boards
                    ea_df["line_name_am"]  = ea_df["line_name"].str.replace('EA','AM')

                    path_boardings = self.get_abs_path(self.controller.config.transit.output_transit_boardings_path)
                    ea_df.to_csv(path_boardings.format(period='ea_pnr'), index=False)
       
                if self.controller.config.transit.get("output_transit_boardings_path"):
                    self.export_boardings_by_line(scenario, period, use_fares)
                if self.controller.config.transit.get("output_shapefile_path"):
                    emme_app.data_explorer().replace_primary_scenario(scenario)
                    self.export_segment_shapefile(emme_app, period)
                if self.controller.config.transit.get("output_stop_usage_path"):
                    self.export_connector_flows(scenario, period)

    @_context
    def _setup(self, scenario, period):
        with self.logger.log_start_end(f"period {period.name}"):
            with self.controller.emme_manager.logbook_trace(f"Transit assignments for period {period.name}"):
                self._matrix_cache = _emme_tools.matrix.MatrixCache(scenario)
                self._skim_matrices = []	
                try:
                    yield
                finally:
                    self._matrix_cache.clear()
                    self._matrix_cache = None

    @LogStartEnd("prepare network attributes and update times from auto network")
    def update_auto_times(self, transit_network, period):
        if not os.path.isabs(self.controller.config.emme.highway_database_path):
            auto_emmebank_path = self.get_abs_path(self.controller.config.emme.highway_database_path)
        auto_emmebank = self.controller.emme_manager.emmebank(auto_emmebank_path)
        auto_scenario = auto_emmebank.scenario(period.emme_scenario_id)
        if auto_scenario.has_traffic_results:
            # TODO: partial network load
            auto_network = auto_scenario.get_network()
            link_lookup = {}
            for auto_link in auto_network.links():
                link_lookup[auto_link["#link_id"]] = auto_link
            for tran_link in transit_network.links():
                auto_link = link_lookup.get(tran_link["#link_id"])
                if not auto_link:
                    continue
                # TODO: may need to remove "reliability" factor in future versions of VDF definition
                auto_time = auto_link.auto_time
                area_type = auto_link['@area_type']
                if auto_time > 0:
                    # https://github.com/BayAreaMetro/travel-model-one/blob/master/model-files/scripts/skims/PrepHwyNet.job#L106
                    tran_speed = 60 * tran_link.length/auto_time
                    if (tran_link['@ft']<=4 or tran_link['@ft']==8) and (tran_speed<6):
                        tran_speed = 6
                        tran_link["@trantime"] = 60 * tran_link.length/tran_speed
                    elif (tran_speed<3):
                        tran_speed = 3
                        tran_link["@trantime"] = 60 * tran_link.length/tran_speed
                    else:
                        tran_link["@trantime"] = auto_time
                    tran_link.data1 = tran_link["@trantime"] # used in Mixed-Mode transit assigment
                # add bus time calculation below
                    if tran_link["@ft"] in [1,2,3,8]:
                        delayfactor = 0.0
                    else:
                        if area_type in [0,1]: 
                            delayfactor = 2.46
                        elif area_type in [2,3]: 
                            delayfactor = 1.74
                        elif area_type==4:
                            delayfactor = 1.14
                        else:
                            delayfactor = 0.08
                    bus_time = tran_link["@trantime"] + (delayfactor * tran_link.length)
                    tran_link["@trantime"] = bus_time                   

        # set us1 (segment data1), used in ttf expressions, from @trantime
        for segment in transit_network.transit_segments():
            if segment['@schedule_time'] <= 0 and segment.link is not None:
                segment.data1 = segment["@trantime_seg"] = segment.link["@trantime"]

    def update_pnr_penalty(self, network, deflator):
        for segment in network.transit_segments():
            if "BART_acc" in segment.id:
                if "West Oakland" in segment.id:
                    segment["@board_cost"] = 12.4*deflator
                elif "Glen Park" in segment.id:
                    segment["@board_cost"] = 13.0*deflator
                elif "Lake Merritt" in segment.id:
                    segment["@board_cost"] = 11.0*deflator
                elif "San Bruno" in segment.id:
                    segment["@board_cost"] = 4.5*deflator
                elif "Fruitvale" in segment.id:
                    segment["@board_cost"] = 4.5*deflator             
                else:
                    segment["@board_cost"] = 3.0*deflator
            elif "Caltrain_acc" in segment.id:
                segment["@board_cost"] = 5.5*deflator

    def update_connector_times(self, scenario, network, period):
        params = self.controller.config.transit
        connector_attrs = {1:"@access_time", 2:"@access_pfactor"}
        for attr_name in connector_attrs.values():
            if scenario.extra_attribute(attr_name) is None:
                scenario.create_extra_attribute("LINK", attr_name)
            # delete attribute in network object to reinitialize to default values
            if attr_name in network.attributes("LINK"):
                network.delete_attribute("LINK", attr_name)
            network.create_attribute("LINK", attr_name, 9999)

        for link in network.links():
            if link.modes.intersection(set([network.mode('a'), network.mode('e')])):
                link['@access_time'] = 60 * link.length/3
                link['@access_pfactor'] = params["walk_perception_factor"]
            elif link.modes.intersection(set([network.mode('K'), network.mode('P')])):
                link['@access_time'] = 60 * link.length/40
                link['@access_pfactor'] = params["drive_perception_factor"]
            else:
                link['@access_pfactor'] = 1


    @LogStartEnd("initialize matrices")
    def initialize_skim_matrices(self, time_periods, scenario):
        with self.controller.emme_manager.logbook_trace("Create and initialize matrices"):
            tmplt_matrices = [
                ("IWAIT", "first wait time"),
                ("XWAIT", "transfer wait time"),
                ("WAIT", "total wait time"),
                ("FARE", "fare"),
                ("BOARDS", "num boardings"),
                ("WAUX", "auxiliary walk time"),
                ("DTIME", "access and egress drive time"),
                ("DDIST", "access and egress drive distance"),
                ("WACC", "access walk time"),
                ("WEGR", "egress walk time"),
                ("IVT", "total in-vehicle time"),
                ("IVTLOC", "local bus in-vehicle time"),
                ("IVTEXP", "express bus in-vehicle time"),
                ("IVTLRT", "light rail in-vehicle time"),
                ("IVTHVY", "heavy rail in-vehicle time"),
                ("IVTCOM", "commuter rail in-vehicle time"),
                ("IVTFRY", "ferry in-vehicle time"),
                ("IN_VEHICLE_COST", "In vehicle cost"),  
                ("CROWD", "Crowding penalty"),
            ]
            skim_sets = [
                ("PNR_TRN_WLK", "PNR access"),
                ("WLK_TRN_PNR", "PNR egress"),
                ("KNR_TRN_WLK", "KNR access"),
                ("WLK_TRN_KNR", "KNR egress"),
                ("WLK_TRN_WLK", "Walk access"),
            ]
            matrices = [("ms", "zero", "zero")]
            emmebank = scenario.emmebank
            for period in time_periods:
                for set_name, set_desc in skim_sets:
                    for name, desc in tmplt_matrices:
                        matrices.append(("mf", f"{period}_{set_name}_{name}", f"{period} {set_desc}: {desc}"))
            # check on database dimensions
            dim_full_matrices = emmebank.dimensions["full_matrices"]
            used_matrices = len([m for m in emmebank.matrices() if m.type == "FULL"])
            if len(matrices) > dim_full_matrices - used_matrices:
                raise Exception(
                    "emmebank full_matrix capacity insuffcient, increase to at least %s"
                    % (len(matrices) + used_matrices)
                )
            create_matrix = self.controller.emme_manager.tool("inro.emme.data.matrix.create_matrix")
            for mtype, name, desc in matrices:
                matrix = emmebank.matrix(f'{mtype}"{name}"')
                if matrix:
                    emmebank.delete_matrix(matrix)
                create_matrix(mtype, name, desc, scenario=scenario, overwrite=True)

    @LogStartEnd("Import transit demand")
    def import_demand_matrices(self, period_name, scenario):
        # TODO: this needs some work
        #      - would like to save multiple matrices per OMX file (requires CT-RAMP changes)
        #      - would like to cross-reference the transit class structure 
        #        and the demand grouping in the config (identical to highway)
        #      - period should be the only placeholder key
        #      - should use separate methods to load and sum the demand
        #      - matrix names

        num_zones = len(scenario.zone_numbers)
        emmebank = scenario.emmebank
        msa_iteration = self.controller.iteration
        omx_filename_template = os.path.join(self.controller.config.household.transit_demand_file)
        matrix_name_template = "{access_mode}"
        emme_matrix_name_template = "{access_mode}_{period}"  # Consolidate LOCAL, PREM, ALLPEN
        # with _m.logbook_trace("Importing demand matrices for period %s" % period):
        
        omx_filename_path = self.get_abs_path(
            omx_filename_template.format(
                period=period_name
            )
        )
        with OMXManager(omx_filename_path) as file_obj:
            for access_mode in _all_access_modes:
                matrix_name = matrix_name_template.format(access_mode=access_mode)
                demand = file_obj.read(matrix_name.upper())  
                shape = demand.shape
                # pad external zone values with 0
                if shape != (num_zones, num_zones):
                    demand = np.pad(
                        demand, ((0, num_zones - shape[0]), (0, num_zones - shape[1]))
                    )
                demand_name = emme_matrix_name_template.format(period=period_name, access_mode=access_mode)
                matrix = emmebank.matrix(f'mf"{demand_name}"')
                apply_msa_demand = self.controller.config.transit.get("apply_msa_demand")
                if msa_iteration <= 1:
                    if not matrix:
                        ident = emmebank.available_matrix_identifier("FULL")
                        matrix = emmebank.create_matrix(ident)
                        matrix.name = demand_name
                    # matrix.description = ?
                elif apply_msa_demand:
                    # Load prev demand and MSA average
                    prev_demand = matrix.get_numpy_data(scenario.id)
                    demand = prev_demand + (1.0 / msa_iteration) * (
                            demand - prev_demand
                    )
                matrix.set_numpy_data(demand, scenario.id)

    def create_empty_demand_matrices(self, period_name, scenario):
        emme_matrix_name_template = "{access_mode}_{period}"
        emmebank = scenario.emmebank
        for access_mode in _all_access_modes:
            demand_name = emme_matrix_name_template.format(period=period_name, access_mode=access_mode)
            matrix = emmebank.matrix(demand_name)
            if not matrix:
                ident = emmebank.available_matrix_identifier("FULL")
                matrix = emmebank.create_matrix(ident)
                matrix.name = demand_name
            else:
                matrix.initialize(0)
            matrix.description = f"{period_name} {access_mode}"[:80]

    def assign_and_skim(self,
                        scenario,
                        network,
                        period,
                        assignment_only=False,
                        use_fares=False,
                        use_ccr=False,
                        congested_transit_assignment=False
                        ):
        # TODO: double check value of time from $/min to $/hour is OK
        # network = scenario.get_network()
        # network = scenario.get_partial_network(
        #     element_types=["TRANSIT_LINE", "TRANSIT_SEGMENT"], include_attributes=True)
        mode_types = {"TRN": [], "WALK": [], "PNR_ACCESS": [], "PNR_EGRESS": [],"KNR_ACCESS": [],"KNR_EGRESS": []}
        for mode in self.controller.config.transit.modes:
            if mode.type in ["WALK"]:
                mode_types["WALK"].append(mode.mode_id)
                mode_types["PNR_ACCESS"].append(mode.mode_id)
                mode_types["PNR_EGRESS"].append(mode.mode_id)
                mode_types["KNR_ACCESS"].append(mode.mode_id)
                mode_types["KNR_EGRESS"].append(mode.mode_id)
            elif mode.type in ["ACCESS"]:
                mode_types["WALK"].append(mode.mode_id)
                mode_types["PNR_EGRESS"].append(mode.mode_id)   # walk access + PNR egress
                mode_types["KNR_EGRESS"].append(mode.mode_id)   # walk access + KNR egress
            elif mode.type in ["EGRESS"]:
                mode_types["WALK"].append(mode.mode_id)
                mode_types["PNR_ACCESS"].append(mode.mode_id)   # PNR access + walk egress
                mode_types["KNR_ACCESS"].append(mode.mode_id)   # KNR access + walk egress
            elif mode.type in ["DRIVE"]:
                mode_types["PNR_ACCESS"].append(mode.mode_id)
                mode_types["KNR_ACCESS"].append(mode.mode_id)
                mode_types["PNR_EGRESS"].append(mode.mode_id)
                mode_types["KNR_EGRESS"].append(mode.mode_id)
            elif mode.type in ["KNR_dummy"]:
                mode_types["KNR_ACCESS"].append(mode.mode_id)
                mode_types["KNR_EGRESS"].append(mode.mode_id)
            elif mode.type in ["LOCAL","PREMIUM","PNR_dummy"]:
                mode_types["TRN"].append(mode.mode_id)       
        print(mode_types)            
        with self.controller.emme_manager.logbook_trace("Transit assignment and skims for period %s" % period.name):
            self.run_assignment(
                scenario,
                period,
                network,
                mode_types,
                use_fares,
                use_ccr,
                congested_transit_assignment
            )

            if not assignment_only:
                with self.controller.emme_manager.logbook_trace("Skims for PNR_TRN_WLK"):
                    self.run_skims(
                        scenario,
                        "PNR_TRN_WLK",
                        period,
                        mode_types["TRN"],
                        network,
                        use_fares,
                        use_ccr,
                        congested_transit_assignment
                    )
                with self.controller.emme_manager.logbook_trace("Skims for WLK_TRN_PNR"):
                    self.run_skims(
                        scenario,
                        "WLK_TRN_PNR",
                        period,
                        mode_types["TRN"],
                        network,
                        use_fares,
                        use_ccr,
                        congested_transit_assignment                    
                    )
                with self.controller.emme_manager.logbook_trace("Skims for KNR_TRN_WLK"):
                    self.run_skims(
                        scenario,
                        "KNR_TRN_WLK",
                        period,
                        mode_types["TRN"],
                        network,
                        use_fares,
                        use_ccr,
                        congested_transit_assignment                        
                    )
                with self.controller.emme_manager.logbook_trace("Skims for WLK_TRN_KNR"):
                    self.run_skims(
                        scenario,
                        "WLK_TRN_KNR",
                        period,
                        mode_types["TRN"],
                        network,
                        use_fares,
                        use_ccr,
                        congested_transit_assignment                        
                    )
                with self.controller.emme_manager.logbook_trace("Skims for WLK_TRN_WLK"):
                    self.run_skims(
                        scenario,
                        "WLK_TRN_WLK",
                        period,
                        mode_types["TRN"],
                        network,
                        use_fares,
                        use_ccr,
                        congested_transit_assignment                       
                    )
                    if self.controller.config.transit.get("mask_noncombo_allpen", True):
                        self.mask_allpen(period.name)
                if self.controller.config.transit.get("mask_over_3_xfers", True):
                    self.mask_transfers(period.name)
                # report(scenario, period)

    def run_assignment(
            self,
            scenario,
            period,
            network,
            mode_types,
            use_fares=False,
            use_ccr=False,
            congested_transit_assignment=False
    ):

        # REVIEW: separate method into smaller steps
        #     - specify class structure in config
        #     - 
        params = self.controller.config.transit
        modeller = self.controller.emme_manager.modeller()
        base_spec = {
            "type": "EXTENDED_TRANSIT_ASSIGNMENT",
            "modes": [],
            "demand": "",  # demand matrix specified below
            "waiting_time": {
                "effective_headways": params["effective_headway_source"],
                "headway_fraction": "@hdw_fraction",
                "perception_factor": params["initial_wait_perception_factor"],
                "spread_factor": 1.0,
            },
            "boarding_cost": {"global": {"penalty": 0, "perception_factor": 1}},
            "boarding_time": {"on_lines": {
                "penalty": "@iboard_penalty", "perception_factor": 1}
            },
            "in_vehicle_cost": None,
            "in_vehicle_time": {"perception_factor": "@invehicle_factor"},
            "aux_transit_time": {"perception_factor": 1}, # walk and drive perception factors are specified in mode speed
            "aux_transit_cost": None,
            "journey_levels": [],
            "flow_distribution_between_lines": {"consider_total_impedance": False},
            "flow_distribution_at_origins": {
                "fixed_proportions_on_connectors": None,
                "choices_at_origins": "OPTIMAL_STRATEGY",
            },
            "flow_distribution_at_regular_nodes_with_aux_transit_choices": {
                "choices_at_regular_nodes": "OPTIMAL_STRATEGY"
            },
            "circular_lines": {"stay": False},
            "connector_to_connector_path_prohibition": None,
            "od_results": {"total_impedance": None},
            "performance_settings": {"number_of_processors": self._num_processors},
        }
        if use_fares:
            # fare attributes
            fare_perception = 60 / params["value_of_time"]
            base_spec["boarding_cost"] = {
                "on_segments": {
                    "penalty": "@board_cost",
                    "perception_factor": fare_perception,
                }
            }
            base_spec["in_vehicle_cost"] = {
                "penalty": "@invehicle_cost",
                "perception_factor": fare_perception,
            }

            fare_modes = _defaultdict(lambda: set([]))
            for line in network.transit_lines():
                fare_modes[line["#src_mode"]].add(line.mode.id)

            def get_fare_modes(src_modes):
                out_modes = set([])
                for mode in src_modes:
                    out_modes.update(fare_modes[mode])
                return list(out_modes)

            all_modes = get_fare_modes(mode_types["TRN"])
            project_dir = os.path.dirname(os.path.dirname(scenario.emmebank.path))

            PNR_TRN_WLK_journey_levels = update_journey_levels_with_fare(
                project_dir, 
                period, 
                "PNR_TRN_WLK", 
                fare_perception, 
                params
            )
            WLK_TRN_PNR_journey_levels = update_journey_levels_with_fare(
                project_dir, 
                period, 
                "WLK_TRN_PNR", 
                fare_perception, 
                params
            )
            KNR_TRN_WLK_journey_levels = update_journey_levels_with_fare(
                project_dir, 
                period, 
                "KNR_TRN_WLK", 
                fare_perception, 
                params
            )
            WLK_TRN_KNR_journey_levels = update_journey_levels_with_fare(
                project_dir, 
                period, 
                "WLK_TRN_KNR", 
                fare_perception, 
                params
            )
            WLK_TRN_WLK_journey_levels = update_journey_levels_with_fare(
                project_dir, 
                period, 
                "WLK_TRN_WLK", 
                fare_perception, 
                params
            )
            mode_attr = '["#src_mode"]'
        else:
            all_modes = list(mode_types["TRN"])
            journey_levels = get_jl_xfer_penalty(
                all_modes,
                params["effective_headway_source"],
                params["transfer_wait_perception_factor"],
                "@xboard_penalty"
            )
            WLK_TRN_WLK_journey_levels = journey_levels
            PNR_TRN_WLK_journey_levels = journey_levels
            WLK_TRN_PNR_journey_levels = journey_levels
            KNR_TRN_WLK_journey_levels = journey_levels
            WLK_TRN_KNR_journey_levels = journey_levels
            mode_attr = ".mode.mode_id"
        print(all_modes)
        skim_parameters = OrderedDict(
            [
                (
                    "WLK_TRN_WLK",
                    {
                        "modes": mode_types["WALK"] + all_modes,
                        "journey_levels": WLK_TRN_WLK_journey_levels,
                    },
                ),
                (
                    "PNR_TRN_WLK",
                    {
                        "modes": mode_types["PNR_ACCESS"] + all_modes,
                        "journey_levels": PNR_TRN_WLK_journey_levels,
                    },
                ),
                (
                    "WLK_TRN_PNR",
                    {
                        "modes": mode_types["PNR_EGRESS"] + all_modes,
                        "journey_levels": WLK_TRN_PNR_journey_levels,
                    },
                ),
                (
                    "KNR_TRN_WLK",
                    {
                        "modes": mode_types["KNR_ACCESS"] + all_modes,
                        "journey_levels": KNR_TRN_WLK_journey_levels,
                    },
                ),
                (
                    "WLK_TRN_KNR",
                    {
                        "modes": mode_types["KNR_EGRESS"] + all_modes,
                        "journey_levels": WLK_TRN_KNR_journey_levels,
                    },
                ),
            ]
        )
        if self.controller.config.transit.get("override_connector_times", False):
            skim_parameters["WLK_TRN_WLK"]["aux_transit_cost"] = {
                "penalty": "@access_time", "perception_factor": "@access_pfactor"
            }
            skim_parameters["PNR_TRN_WLK"]["aux_transit_cost"] = {
                "penalty": "@access_time", "perception_factor": "@access_pfactor"
            }
            skim_parameters["WLK_TRN_PNR"]["aux_transit_cost"] = {
                "penalty": "@access_time", "perception_factor": "@access_pfactor"
            }
            skim_parameters["KNR_TRN_WLK"]["aux_transit_cost"] = {
                "penalty": "@access_time", "perception_factor": "@access_pfactor"
            }
            skim_parameters["WLK_TRN_KNR"]["aux_transit_cost"] = {
                "penalty": "@access_time", "perception_factor": "@access_pfactor"
            }
        if use_ccr:
            print('run capacitated transit assignment')
            assign_transit = modeller.tool(
                "inro.emme.transit_assignment.capacitated_transit_assignment"
            )
            #  assign all 3 classes of demand at the same time
            specs = []
            names = []
            demand_matrix_template = "mf{access_mode_set}_{period}"
            for mode_name, parameters in skim_parameters.items():
                spec = _copy(base_spec)
                spec["modes"] = parameters["modes"]
                demand_matrix = demand_matrix_template.format(
                    access_mode_set=mode_name, period=period.name
                )
                # TODO: need to raise on zero demand matrix?
                # if emmebank.matrix(demand_matrix).get_numpy_data(scenario.id).sum() == 0:
                #     continue  # don't include if no demand
                spec["demand"] = demand_matrix
                spec["journey_levels"] = parameters["journey_levels"]
                # Optional aux_transit_cost, used for walk time on connectors, set if override_connector_times
                spec["aux_transit_cost"] = parameters.get("aux_transit_cost")
                specs.append(spec)
                names.append(mode_name)
            func = {
                "segment": {
                    "type": "CUSTOM",
                    "python_function": _segment_cost_function.format(
                        period.duration
                    ),
                    "congestion_attribute": "us3",
                    "orig_func": False,
                },
                "headway": {
                    "type": "CUSTOM",
                    "python_function": _headway_cost_function.format(mode_attr),
                },
                "assignment_period": period.duration,
            }
            stop = {
                "max_iterations": 10,
                "relative_difference": 0.01,
                "percent_segments_over_capacity": 0.01,
            }

            assign_transit(
                specs,
                congestion_function=func,
                stopping_criteria=stop,
                class_names=names,
                scenario=scenario,
                log_worksheets=False,
            )
        elif congested_transit_assignment:
            print('run congested transit assignment')
            assign_transit = modeller.tool(
                "inro.emme.transit_assignment.congested_transit_assignment"
            )
            #  assign all 3 classes of demand at the same time
            specs = []
            names = []
            demand_matrix_template = "mf{access_mode_set}_{period}"
            for mode_name, parameters in skim_parameters.items():
                spec = _copy(base_spec)
                spec["modes"] = parameters["modes"]
                demand_matrix = demand_matrix_template.format(
                    access_mode_set=mode_name, period=period.name
                )
                # TODO: need to raise on zero demand matrix?
                # if emmebank.matrix(demand_matrix).get_numpy_data(scenario.id).sum() == 0:
                #     continue  # don't include if no demand
                spec["demand"] = demand_matrix
                spec["journey_levels"] = parameters["journey_levels"]
                # Optional aux_transit_cost, used for walk time on connectors, set if override_connector_times
                spec["aux_transit_cost"] = parameters.get("aux_transit_cost")
                specs.append(spec)
                names.append(mode_name)
            # func = {
            #     "type": "BPR",
            #     "weight": 0.15,
            #     "exponent": 4,
            #     "assignment_period": period.length_hours,
            #     "orig_func": False,
            #     "congestion_attribute": "us3"
            # }
            func = {
                "type": "CUSTOM",
                "python_function": _segment_cost_function.format(period.length_hours, mode_attr),
                "congestion_attribute": "us3",
                "orig_func": False,
                "assignment_period": period.length_hours,
            }
            stop = {
                "max_iterations": 10,
                "normalized_gap": 0.01,
                "relative_gap": 0.001
            }

            assign_transit(
                specs,
                congestion_function=func,
                stopping_criteria=stop,
                class_names=names,
                scenario=scenario,
                log_worksheets=False,
            )
        else:
            print('run extended transit assignment')
            assign_transit = modeller.tool(
                "inro.emme.transit_assignment.extended_transit_assignment"
            )
            add_volumes = False
            for mode_name, parameters in skim_parameters.items():
                spec = _copy(base_spec)
                spec["modes"] = parameters["modes"]
                # spec["demand"] = 'ms1' # zero demand matrix
                spec["demand"] = "mf{access_mode_set}_{period}".format(
                    access_mode_set=mode_name, period=period.name
                )
                spec["journey_levels"] = parameters["journey_levels"]
                # Optional aux_transit_cost, used for walk time on connectors, set if override_connector_times
                spec["aux_transit_cost"] = parameters.get("aux_transit_cost")
                assign_transit(
                    spec, class_name=mode_name, add_volumes=add_volumes, scenario=scenario
                )
                add_volumes = True

    def run_skims(self,
                  scenario,
                  name,
                  period,
                  valid_modes,
                  network,
                  use_fares=False,
                  use_ccr=False,
                  congested_transit_assignment=False
                  ):
        # REVIEW: separate method into smaller steps
        #     - specify class structure in config
        #     - specify skims by name
        modeller = self.controller.emme_manager.modeller()
        num_processors = self._num_processors
        matrix_calc = modeller.tool("inro.emme.matrix_calculation.matrix_calculator")
        network_calc = modeller.tool("inro.emme.network_calculation.network_calculator")
        create_extra = modeller.tool(
            "inro.emme.data.extra_attribute.create_extra_attribute"
        )
        matrix_results = modeller.tool(
            "inro.emme.transit_assignment.extended.matrix_results"
        )
        path_analysis = modeller.tool(
            "inro.emme.transit_assignment.extended.path_based_analysis"
        )
        strategy_analysis = modeller.tool(
            "inro.emme.transit_assignment.extended.strategy_based_analysis"
        )
 
        override_connectors = self.controller.config.transit.get("override_connector_times", False)
        class_name = name
        skim_name = "%s_%s" % (period.name, name)
        with self.controller.emme_manager.logbook_trace(
                "First and total wait time, number of boardings, fares, total walk time"
        ):
            # First and total wait time, number of boardings, fares, total walk time, in-vehicle time
            spec = {
                "type": "EXTENDED_TRANSIT_MATRIX_RESULTS",
                "actual_first_waiting_times": 'mf"%s_IWAIT"' % skim_name,
                "actual_total_waiting_times": 'mf"%s_WAIT"' % skim_name,
                "by_mode_subset": {
                    "modes": [
                        m.id
                        for m in network.modes()
                        if m.type in ["TRANSIT", "AUX_TRANSIT"]
                    ],
                    "avg_boardings": 'mf"%s_BOARDS"' % skim_name,
                },
            }
            if use_fares:
                spec["by_mode_subset"]["actual_in_vehicle_costs"] = (
                        'mf"%s_IN_VEHICLE_COST"' % skim_name
                )
                spec["by_mode_subset"]["actual_total_boarding_costs"] = (
                        'mf"%s_FARE"' % skim_name
                )
            matrix_results(
                spec,
                class_name=class_name,
                scenario=scenario,
                num_processors=num_processors,
            )
            # xfer_modes = []
            # for mode in self.controller.config.transit.modes:
            #     if mode.type == "WALK":
            #         xfer_modes.append(mode.mode_id)
            spec = {
                "type": "EXTENDED_TRANSIT_MATRIX_RESULTS",
                "by_mode_subset": {"modes": ["w"], "actual_aux_transit_times": 'mf"%s_WAUX"' % skim_name}, #https://github.com/BayAreaMetro/modeling-website/wiki/TransitSkims
            }
            spec1 = {
                "type": "EXTENDED_TRANSIT_MATRIX_RESULTS",
                "by_mode_subset": {"modes": ["D"], 
                "actual_aux_transit_times": 'mf"%s_DTIME"' % skim_name,
                "distance": 'mf"%s_DDIST"' % skim_name},
            }
            spec2 = {
                "type": "EXTENDED_TRANSIT_MATRIX_RESULTS",
                "by_mode_subset": {"modes": ["a"], "actual_aux_transit_times": 'mf"%s_WACC"' % skim_name},
            }
            spec3 = {
                "type": "EXTENDED_TRANSIT_MATRIX_RESULTS",
                "by_mode_subset": {"modes": ["e"], "actual_aux_transit_times": 'mf"%s_WEGR"' % skim_name},
            }
            matrix_results(
                spec,
                class_name=class_name,
                scenario=scenario,
                num_processors=num_processors,
            )

            matrix_results(
                spec1,
                class_name=class_name,
                scenario=scenario,
                num_processors=num_processors,
            )

            matrix_results(
                spec2,
                class_name=class_name,
                scenario=scenario,
                num_processors=num_processors,
            )

            matrix_results(
                spec3,
                class_name=class_name,
                scenario=scenario,
                num_processors=num_processors,
            )

        with self.controller.emme_manager.logbook_trace("In-vehicle time by mode"):
            mode_combinations = [
                ("LOC", "b"),
                ("EXP", "x"),
                ("LRT", "l"),
                ("HVY", "h"),
                ("COM", "r"),
                ("FRY", "f"),
            ]
            # map to used modes in apply fares case
            fare_modes = _defaultdict(lambda: set([]))
            if use_fares:
                for line in network.transit_lines():
                    fare_modes[line["#src_mode"]].add(line.mode.id)
            else:
                fare_modes = dict((m, [m]) for m in valid_modes)
            # set to fare_modes and filter out unused modes
            mode_combinations = [
                (n, list(fare_modes[m])) for n, m in mode_combinations if m in valid_modes
            ]

            total_ivtt_expr = []
            if use_ccr:
                mode_combinations = [
                ("LOC", "b"),
                ("EXP", "x"),
                ("LRT", "l"),
                ("HVY", "h"),
                ("COM", "r"),
                ("FRY", "f"),
                ]
                scenario.create_extra_attribute("TRANSIT_SEGMENT", "@mode_timtr")
                try:
                    for mode_name, modes in mode_combinations:
                        network.create_attribute("TRANSIT_SEGMENT", "@mode_timtr")
                        for line in network.transit_lines():
                            # if line.mode.id in modes:
                            if line['#src_mode'] in modes:
                                for segment in line.segments():
                                    # segment["@mode_timtr"] = segment["@base_timtr"]
                                    # segment["@mode_timtr"] = segment["@trantime_final"]
                                    segment["@mode_timtr"] = segment["transit_time"]
                        mode_timtr = network.get_attribute_values(
                            "TRANSIT_SEGMENT", ["@mode_timtr"]
                        )
                        network.delete_attribute("TRANSIT_SEGMENT", "@mode_timtr")
                        scenario.set_attribute_values(
                            "TRANSIT_SEGMENT", ["@mode_timtr"], mode_timtr
                        )
                        ivtt = 'mf"%s_IVT%s"' % (skim_name, mode_name)
                        total_ivtt_expr.append(ivtt)
                        spec = get_strat_spec({"in_vehicle": "@mode_timtr"}, ivtt)
                        strategy_analysis(
                            spec,
                            class_name=class_name,
                            scenario=scenario,
                            num_processors=num_processors,
                        )

                finally:
                    scenario.delete_extra_attribute("@mode_timtr")
            else:
                for mode_name, modes in mode_combinations:
                    ivtt = 'mf"%s_IVT%s"' % (skim_name, mode_name)
                    total_ivtt_expr.append(ivtt)
                    spec = {
                        "type": "EXTENDED_TRANSIT_MATRIX_RESULTS",
                        "by_mode_subset": {"modes": modes, 
                                        "actual_in_vehicle_times": ivtt,
                        },
                    }
                    matrix_results(
                        spec,
                        class_name=class_name,
                        scenario=scenario,
                        num_processors=num_processors,
                    )

        with self.controller.emme_manager.logbook_trace(
                "Calculate total IVTT, number of transfers, transfer walk and wait times"
        ):
            spec_list = [
                {  # sum total ivtt across all modes
                    "type": "MATRIX_CALCULATION",
                    "constraint": None,
                    "result": f'mf"{skim_name}_IVT"',
                    "expression": "+".join(total_ivtt_expr),
                },
                {
                    "type": "MATRIX_CALCULATION",
                    "constraint": {
                        "by_value": {
                            "od_values": f'mf"{skim_name}_WAIT"',
                            "interval_min": 0,
                            "interval_max": 9999999,
                            "condition": "INCLUDE",
                        }
                    },
                    "result": f'mf"{skim_name}_XWAIT"',
                    "expression": f'(mf"{skim_name}_WAIT" - mf"{skim_name}_IWAIT").max.0',
                },
            ]
            if use_fares:
                # sum in-vehicle cost and boarding cost to get the fare paid
                spec_list.append({
                    "type": "MATRIX_CALCULATION",
                    "constraint": None,
                    "result": f'mf"{skim_name}_FARE"',
                    "expression": f'(mf"{skim_name}_FARE" + mf"{skim_name}_IN_VEHICLE_COST")'})
            if ("PNR_TRN_WLK" in skim_name) or ("WLK_TRN_PNR"in skim_name):
                spec_list.append(                
                {  # subtract PNR boarding from total boardings
                    "type": "MATRIX_CALCULATION",
                    "constraint": {
                        "by_value": {
                            "od_values": f'mf"{skim_name}_BOARDS"',
                            "interval_min": 0,
                            "interval_max": 9999999,
                            "condition": "INCLUDE",
                        }
                    },
                    "result": f'mf"{skim_name}_BOARDS"',
                    "expression": f'(mf"{skim_name}_BOARDS" - 1).max.0',
                })                

            matrix_calc(spec_list, scenario=scenario, num_processors=num_processors)

        if use_ccr:
            with self.controller.emme_manager.logbook_trace("Calculate CCR skims"):
                create_extra(
                    "TRANSIT_SEGMENT",
                    "@eawt",
                    "extra added wait time",
                    overwrite=True,
                    scenario=scenario,
                )
                # create_extra("TRANSIT_SEGMENT", "@crowding_factor",
                # "crowding factor along segments", overwrite=True, scenario=scenario)
                create_extra(
                    "TRANSIT_SEGMENT",
                    "@capacity_penalty",
                    "capacity penalty at boarding",
                    overwrite=True,
                    scenario=scenario,
                )
                network = scenario.get_partial_network(
                    ["TRANSIT_LINE", "TRANSIT_SEGMENT"], include_attributes=True
                )
                attr_map = {
                    "TRANSIT_SEGMENT": ["@phdwy", "transit_volume", "transit_boardings"],
                    "TRANSIT_VEHICLE": ["seated_capacity", "total_capacity"],
                    "TRANSIT_LINE": ["headway"],
                }
                if use_fares:
                    # only if use_fares, otherwise will use .mode.id
                    attr_map["TRANSIT_LINE"].append("#src_mode")
                    mode_name = '["#src_mode"]'
                else:
                    mode_name = ".mode.id"
                for domain, attrs in attr_map.items():
                    values = scenario.get_attribute_values(domain, attrs)
                    network.set_attribute_values(domain, attrs, values)

                enclosing_scope = {"network": network, "scenario": scenario}
                # code = compile(_segment_cost_function, "segment_cost_function", "exec")
                # exec(code, enclosing_scope)
                code = compile(
                    _headway_cost_function.format(mode_name),
                    "headway_cost_function",
                    "exec",
                )
                exec(code, enclosing_scope)
                calc_eawt = enclosing_scope["calc_eawt"]
                hdwy_fraction = 0.5  # fixed in assignment spec

                for segment in network.transit_segments():
                    headway = segment.line.headway
                    veh_cap = line.vehicle.total_capacity
                    # capacity = 60.0 * veh_cap / line.headway
                    capacity = 60.0 * period.duration * veh_cap / line.headway
                    transit_volume = segment.transit_volume
                    vcr = transit_volume / capacity
                    segment["@eawt"] = calc_eawt(segment, vcr, headway)
                    # segment["@crowding_penalty"] = calc_segment_cost(transit_volume, capacity, segment)
                    segment["@capacity_penalty"] = (
                            max(segment["@phdwy"] - segment["@eawt"] - headway, 0)
                            * hdwy_fraction
                    )

                values = network.get_attribute_values(
                    "TRANSIT_SEGMENT", ["@eawt", "@capacity_penalty"]
                )
                scenario.set_attribute_values(
                    "TRANSIT_SEGMENT", ["@eawt", "@capacity_penalty"], values
                )

                # # Link unreliability
                # spec = get_strat_spec({"in_vehicle": "ul1"}, "%s_LINKREL" % skim_name)
                # strategy_analysis(spec, class_name=class_name, scenario=scenario, num_processors=num_processors)

                # Crowding penalty
                spec = get_strat_spec({"in_vehicle": "@ccost"}, f'mf"{skim_name}_CROWD"')
                strategy_analysis(
                    spec,
                    class_name=class_name,
                    scenario=scenario,
                    num_processors=num_processors,
                )

                # skim node reliability, Extra added wait time (EAWT)
                spec = get_strat_spec({"boarding": "@eawt"}, f'mf"{skim_name}_EAWT"')
                strategy_analysis(
                    spec,
                    class_name=class_name,
                    scenario=scenario,
                    num_processors=num_processors,
                )

                # skim capacity penalty
                spec = get_strat_spec(
                    {"boarding": "@capacity_penalty"}, f'mf"{skim_name}_CAPPEN"'
                )
                strategy_analysis(
                    spec,
                    class_name=class_name,
                    scenario=scenario,
                    num_processors=num_processors,
                )

        if congested_transit_assignment:
            spec = get_strat_spec({"in_vehicle": "@ccost"}, f'mf"{skim_name}_CROWD"')
            strategy_analysis(
                spec,
                class_name=class_name,
                scenario=scenario,
                num_processors=num_processors,
            )

    def mask_allpen(self, period):
        # Reset skims to 0 if not both local and premium
        localivt_skim = self._matrix_cache.get_data(f'mf"{period}_ALLPEN_LBIVTT"')
        totalivt_skim = self._matrix_cache.get_data(f'mf"{period}_ALLPEN_TOTALIVTT"')
        has_premium = np.greater((totalivt_skim - localivt_skim), 0)
        has_both = np.greater(localivt_skim, 0) * has_premium
        for skim in _skim_names:
            mat_name = f'mf"{period}_ALLPEN_{skim}"'
            data = self._matrix_cache.get_data(mat_name)
            self._matrix_cache.set_data(mat_name, data * has_both)

    def mask_transfers(self, period):
        # Reset skims to 0 if number of transfers is greater than max_transfers
        max_transfers = self.controller.config.transit.max_transfers
        for skim_set in ["BUS", "PREM", "ALLPEN"]:
            xfers = self._matrix_cache.get_data(f'mf"{period}_{skim_set}_XFERS"')
            xfer_mask = np.less_equal(xfers, max_transfers)
            for skim in _skim_names:
                mat_name = f'mf"{period}_{skim_set}_{skim}"'
                data = self._matrix_cache.get_data(mat_name)
                self._matrix_cache.set_data(mat_name, data * xfer_mask)

    def export_skims(self, period, scenario):
        """Export skims to OMX files by period."""
        # NOTE: skims in separate file by period
        skim_sets = [
            ("PNR_TRN_WLK", "PNR access"),
            ("WLK_TRN_PNR", "PNR egress"),
            ("KNR_TRN_WLK", "KNR access"),
            ("WLK_TRN_KNR", "KNR egress"),
            ("WLK_TRN_WLK", "Walk access"),
        ]
        for set_name, set_desc in skim_sets:
            matrices = {}
            matrices_growth = {} # matrices need to be multiplied by 100

            output_skim_path = self.get_abs_path(
                self.controller.config.transit.output_skim_path
            )
            omx_file_path = os.path.join(
                output_skim_path,
                self.controller.config.transit.output_skim_filename_tmpl.format(time_period=period, set_name=set_name))
            os.makedirs(os.path.dirname(omx_file_path), exist_ok=True)

            for skim in _skim_names:
                if "BOARDS" in skim:
                    matrices[skim] = (f'mf"{period}_{set_name}_{skim}"')
                else:
                    matrices_growth[skim] = (f'mf"{period}_{set_name}_{skim}"')

            with OMXManager(
                    omx_file_path, "w", scenario, matrix_cache=self._matrix_cache, mask_max_value=1e7, growth_factor=1
            ) as omx_file:
                omx_file.write_matrices(matrices)

            with OMXManager(
                    omx_file_path, "a", scenario, matrix_cache=self._matrix_cache, mask_max_value=1e7, growth_factor=100
            ) as omx_file:
                omx_file.write_matrices(matrices_growth)

            self._matrix_cache.clear()

    def export_boardings_by_line(self, scenario, period, use_fares):
        network = scenario.get_network()
        path_boardings = self.get_abs_path(self.controller.config.transit.output_transit_boardings_path)
        with open(path_boardings.format(period=period.name), "w") as f:
            f.write(",".join(["line_name", 
                            "description", 
                            "total_boarding",
                            'total_hour_cap',
                            "tm2_mode", 
                            "line_mode", 
                            "headway", 
                            "fare_system", 
                            ]))
            f.write("\n")

            for line in network.transit_lines():
                boardings = 0
                capacity = line.vehicle.total_capacity
                hdw = line.headway
                line_hour_cap = 60*capacity/hdw
                if use_fares:
                    mode = line['#src_mode']
                else:
                    mode = line.mode
                for segment in line.segments(include_hidden=True):
                    boardings += segment.transit_boardings  
                f.write(",".join([str(x) for x in [line.id, 
                                                line['#description'], 
                                                boardings, 
                                                line_hour_cap,    
                                                line['#mode'], 
                                                mode,
                                                line.headway,
                                                line['#faresystem'],  
                                                ]]))
                f.write("\n")

    def export_segment_shapefile(self, emme_app, period):
        project = emme_app.project
        path_shapefile = self.get_abs_path(self.controller.config.transit.output_shapefile_path)
        table = project.new_network_table("TRANSIT_SEGMENT")
        column = _worksheet.Column()
        column_names = {'line':'line',
                        'i':'i_node',
                        'j':'j_node',
                        'length':'length',
                        'dwt':'dwt',
                        'ttf':'ttf',
                        'voltr':'voltr',
                        'board':'board',
                        'timtr':'con_time',
                        '@trantime_seg':'uncon_time',
                        'mode':'mode',
                        'mdesc':'mdesc',
                        'hdw':'hdw',
                        '@orig_hdw':'orig_hdw',
                        'speed':'speed',
                        'vmode':'vehmode',
                        'vauteq':'vauteq',
                        'vcaps':'vcaps',
                        'vcapt':'vcapt',
                        'caps':'caps',
                        'capt':'capt',
                        'inboa':'inboa',
                        'fiali':'fiali',
                        '#link_id':'#link_id',
                        '@aux_vol_pnr_trn_wlk':'aux_ptw',
                        '@aux_vol_wlk_trn_pnr':'aux_wtp',
                        '@aux_vol_knr_trn_wlk':'aux_ktw',
                        '@aux_vol_wlk_trn_knr':'aux_wtk',
                        '@aux_vol_wlk_trn_wlk':'aux_wtw',
                        }
        i = 0
        for key, item in column_names.items():
            column.expression = key
            column.name = item
            table.add_column(i, column)
            i += 1
        seg_dt = table.save_as_data_table(f"{period.name}_assn", overwrite=True)
        seg_data = seg_dt.get_data()
        filelist = [f for f in os.listdir(os.path.join(path_shapefile.format(period=period.name), "..")) if f.startswith(f"{period.name}_assn")]
        for f in filelist:
            os.remove(os.path.join(path_shapefile.format(period=period.name), "..", f))
        seg_data.export_to_shapefile(path_shapefile.format(period=period.name))
        table.close()

    def export_connector_flows(self, scenario, period):
        # export boardings and alightings by stop (connector) and TAZ
        modeller = self.controller.emme_manager.modeller()
        network_results = modeller.tool(
            "inro.emme.transit_assignment.extended.network_results"
        )
        create_extra = modeller.tool(
            "inro.emme.data.extra_attribute.create_extra_attribute"
        )
        skim_sets = [
            ("PNR_TRN_WLK", "PNR access and Walk egress"),
            ("WLK_TRN_PNR", "Walk access and PNR egress"),
            ("KNR_TRN_WLK", "KNR access and Walk egress"),
            ("WLK_TRN_KNR", "Walk access and KNR egress"),
            ("WLK_TRN_WLK", "Walk access and Walk egress"),
        ]
        names = []
        for name, set_desc in skim_sets:
            attr_name = f"@aux_vol_{name}".lower()
            create_extra("LINK", attr_name, overwrite=True, scenario=scenario)
            spec = {
                "type": "EXTENDED_TRANSIT_NETWORK_RESULTS",
                "on_links": {"aux_transit_volumes": attr_name}
            }
            network_results(spec, class_name=name, scenario=scenario)
            names.append((name, attr_name))

        # TODO: optimization: partial network to only load links and certain attributes
        network = scenario.get_network()
        path_tmplt = self.get_abs_path(self.controller.config.transit.output_stop_usage_path)
        with open(path_tmplt.format(period=period.name), "w") as f:
            f.write(",".join(["mode", "taz", "stop", "boardings", "alightings"]))
            f.write("\n")
            for zone in network.centroids():
                taz_id = int(zone["@taz_id"])
                for link in zone.outgoing_links():
                    stop_id = link.j_node["#node_id"]
                    for name, attr_name in names:
                        boardings = link[attr_name]
                        alightings = link.reverse_link[attr_name] if link.reverse_link else 0.0
                        f.write(",".join([str(x) for x in [name, taz_id, stop_id, boardings, alightings]]))
                        f.write("\n")
                for link in zone.incoming_links():
                    if link.reverse_link:  # already exported
                        continue
                    stop_id = link.i_node["#node_id"]
                    for name, attr_name in names:
                        f.write(",".join([str(x) for x in [name, taz_id, stop_id, 0.0, link[attr_name]]]))
                        f.write("\n")

    def report(self, scenario, period):
        # TODO: untested
        text = ['<div class="preformat">']
        matrices = []
        for skim_set in ["BUS", "PREM", "ALLPEN"]:
            for skim in _skim_names:
                matrices.append(f'mf"{period}_{skim_set}_{skim}"')
        num_zones = len(scenario.zone_numbers)
        num_cells = num_zones * num_zones
        text.append(
            "Number of zones: %s. Number of O-D pairs: %s. "
            "Values outside -9999999, 9999999 are masked in summaries.<br>"
            % (num_zones, num_cells)
        )
        text.append(
            "%-25s %9s %9s %9s %13s %9s" % ("name", "min", "max", "mean", "sum", "mask num")
        )
        for name in matrices:
            data = self._matrix_cache.get_data(name)
            data = np.ma.masked_outside(data, -9999999, 9999999)
            stats = (
                name,
                data.min(),
                data.max(),
                data.mean(),
                data.sum(),
                num_cells - data.count(),
            )
            text.append("%-25s %9.4g %9.4g %9.4g %13.7g %9d" % stats)
        text.append("</div>")
        title = "Transit impedance summary for period %s" % period
        report = _m.PageBuilder(title)
        report.wrap_html("Matrix details", "<br>".join(text))
        self.controller.emme_manager.logbook_write(title, report.render())


def get_jl_xfer_penalty(modes, effective_headway_source, xfer_perception_factor, xfer_boarding_penalty):
    level_rules = [{
        "description": "",
        "destinations_reachable": True,
        "transition_rules": [{"mode": m, "next_journey_level": 1} for m in modes],
    },
        {
            "description": "",
            "destinations_reachable": True,
            "transition_rules": [{"mode": m, "next_journey_level": 1} for m in modes],
            "waiting_time": {
                "headway_fraction": 0.5,
                "effective_headways": effective_headway_source,
                "spread_factor": 1,
                "perception_factor": xfer_perception_factor
            }

        }]

    if xfer_boarding_penalty is not None:
        level_rules[1]["boarding_time"] = {"on_lines": {
            "penalty": xfer_boarding_penalty, "perception_factor": 1}
        }
    return level_rules


def get_strat_spec(components, matrix_name):
    spec = {
        "trip_components": components,
        "sub_path_combination_operator": "+",
        "sub_strategy_combination_operator": "average",
        "selected_demand_and_transit_volumes": {
            "sub_strategies_to_retain": "ALL",
            "selection_threshold": {"lower": -999999, "upper": 999999},
        },
        "analyzed_demand": None,
        "constraint": None,
        "results": {"strategy_values": matrix_name},
        "type": "EXTENDED_TRANSIT_STRATEGY_ANALYSIS",
    }
    return spec


def update_journey_levels_with_fare(project_dir, period, class_name, fare_perception, params):
    with open(
            os.path.join(
                project_dir, "Specifications", "%s_ALLPEN_journey_levels.ems" % period.name
            ),
            "r",
    ) as f:
        journey_levels = _json.load(f)["journey_levels"]

    if class_name == "PNR_TRN_WLK":
        new_journey_levels = copy.deepcopy(journey_levels)

        for i in range(0,len(new_journey_levels)):
            jls = new_journey_levels[i]
            for level in jls["transition_rules"]:
                level["next_journey_level"] = level["next_journey_level"]+1
            jls["transition_rules"].extend(
                [
                {'mode': 'e', 'next_journey_level': i+2},
                {'mode': 'D', 'next_journey_level': len(new_journey_levels)+2}, 
                {'mode': 'w', 'next_journey_level': i+2},
                {'mode': 'p', 'next_journey_level': len(new_journey_levels)+2}
                ]
            )
        
        # level 0: drive access
        transition_rules_drive_access = copy.deepcopy(journey_levels[0]["transition_rules"])
        for level in transition_rules_drive_access:
            level["next_journey_level"] = len(new_journey_levels)+2
        transition_rules_drive_access.extend(
            [
            {'mode': 'e', 'next_journey_level': len(new_journey_levels)+2},
            {'mode': 'D', 'next_journey_level': 0},
            {'mode': 'w', 'next_journey_level': len(new_journey_levels)+2},
            {'mode': 'p', 'next_journey_level': 1}
            ]
        )

        # level 1: use transit
        transition_rules_pnr = copy.deepcopy(journey_levels[0]["transition_rules"])
        for level in transition_rules_pnr:
            level["next_journey_level"] = 2
        transition_rules_pnr.extend(
            [
            {'mode': 'e', 'next_journey_level': len(new_journey_levels)+2},
            {'mode': 'D', 'next_journey_level': len(new_journey_levels)+2},
            {'mode': 'w', 'next_journey_level': len(new_journey_levels)+2},
            {'mode': 'p', 'next_journey_level': 1}
            ]
        )

        # level len(new_journey_levels)+2: every mode is prohibited
        transition_rules_prohibit = copy.deepcopy(journey_levels[0]["transition_rules"])
        for level in transition_rules_prohibit:
            level["next_journey_level"] = len(new_journey_levels)+2
        transition_rules_prohibit.extend(
            [
            {'mode': 'e', 'next_journey_level': len(new_journey_levels)+2},
            {'mode': 'D', 'next_journey_level': len(new_journey_levels)+2},
            {'mode': 'w', 'next_journey_level': len(new_journey_levels)+2},
            {'mode': 'p', 'next_journey_level': len(new_journey_levels)+2}
            ]
        )

        new_journey_levels.insert(
                                0,
                                {
                                "description": "drive access",
                                "destinations_reachable": False,
                                "transition_rules": transition_rules_drive_access,
                                "waiting_time": None,
                                "boarding_time": None,
                                "boarding_cost": None                                     
                                }
        )
        new_journey_levels.insert(
                                1,
                                {
                                "description": "pnr",
                                "destinations_reachable": False,
                                "transition_rules": transition_rules_pnr,
                                "waiting_time": None,
                                "boarding_time": None,
                                "boarding_cost": None                                     
                                }
        )
        new_journey_levels.append(
                                {
                                "description": "prohibit",
                                "destinations_reachable": False,
                                "transition_rules": transition_rules_prohibit,
                                "waiting_time": None,
                                "boarding_time": None,
                                "boarding_cost": None                                     
                                }
        )

        for level in new_journey_levels[2:-1]:
            level["waiting_time"] = {
                "headway_fraction": "@hdw_fraction",
                "effective_headways": params["effective_headway_source"],
                "spread_factor": 1,
                "perception_factor": "@wait_pfactor"
            }
            level["boarding_time"] = {
            "on_lines": {
                "penalty": "@xboard_penalty", "perception_factor": 1},
            "at_nodes": {
                "penalty": "@xboard_nodepen", "perception_factor": 1}, 
            }
        # add in the correct value of time parameter
        for level in new_journey_levels:
            if level["boarding_cost"]:
                level["boarding_cost"]["on_segments"]["perception_factor"] = fare_perception


    elif class_name == "WLK_TRN_PNR":
        new_journey_levels = copy.deepcopy(journey_levels)

        for i in range(0,len(new_journey_levels)):
            jls = new_journey_levels[i]    
            jls["destinations_reachable"] = False
            jls["transition_rules"].extend(
                [
                {'mode': 'a', 'next_journey_level': len(new_journey_levels)+2},
                {'mode': 'D', 'next_journey_level': len(new_journey_levels)+2}, 
                {'mode': 'w', 'next_journey_level': i+1}, 
                {'mode': 'p', 'next_journey_level': len(new_journey_levels)+1}
                ]
            )

         # level 0: walk access
        transition_rules_walk_access = copy.deepcopy(journey_levels[0]["transition_rules"])
        for level in transition_rules_walk_access:
            level["next_journey_level"] = 1
        transition_rules_walk_access.extend(
            [
            {'mode': 'a', 'next_journey_level': 0},
            {'mode': 'D', 'next_journey_level': len(new_journey_levels)+2},
            {'mode': 'w', 'next_journey_level': len(new_journey_levels)+2},
            {'mode': 'p', 'next_journey_level': len(new_journey_levels)+2}
            ]
        )       

        # level len(new_journey_levels)+1: drive home
        transition_rules_drive_home = copy.deepcopy(journey_levels[0]["transition_rules"])
        for level in transition_rules_drive_home:
            level["next_journey_level"] = len(new_journey_levels)+2
        transition_rules_drive_home.extend(
            [
            {'mode': 'a', 'next_journey_level': len(new_journey_levels)+2},
            {'mode': 'D', 'next_journey_level': len(new_journey_levels)+1},
            {'mode': 'w', 'next_journey_level': len(new_journey_levels)+2},
            {'mode': 'p', 'next_journey_level': len(new_journey_levels)+2}
            ]
        )

        # level len(new_journey_levels)+2: every mode is prohibited
        transition_rules_prohibit = copy.deepcopy(journey_levels[0]["transition_rules"])
        for level in transition_rules_prohibit:
            level["next_journey_level"] = len(new_journey_levels)+2
        transition_rules_prohibit.extend(
            [
            {'mode': 'a', 'next_journey_level': len(new_journey_levels)+2},
            {'mode': 'D', 'next_journey_level': len(new_journey_levels)+2},
            {'mode': 'w', 'next_journey_level': len(new_journey_levels)+2},
            {'mode': 'p', 'next_journey_level': len(new_journey_levels)+2}
            ]
        )

        new_journey_levels.insert(
                                0,
                                {
                                "description": "walk access",
                                "destinations_reachable": True,
                                "transition_rules": transition_rules_walk_access,
                                "waiting_time": None,
                                "boarding_time": None,
                                "boarding_cost": None                                     
                                }
        )
        new_journey_levels.append(
                                {
                                "description": "drive home",
                                "destinations_reachable": True,
                                "transition_rules": transition_rules_drive_home,
                                "waiting_time": None,
                                "boarding_time": None,
                                "boarding_cost": None                                     
                                }
        )
        new_journey_levels.append(
                                {
                                "description": "prohibit",
                                "destinations_reachable": False,
                                "transition_rules": transition_rules_prohibit,
                                "waiting_time": None,
                                "boarding_time": None,
                                "boarding_cost": None                                     
                                }
        )

        for level in new_journey_levels[1:-2]:
            level["waiting_time"] = {
                "headway_fraction": "@hdw_fraction",
                "effective_headways": params["effective_headway_source"],
                "spread_factor": 1,
                "perception_factor": "@wait_pfactor"
            }
            level["boarding_time"] = {
            "on_lines": {
                "penalty": "@xboard_penalty", "perception_factor": 1},
            "at_nodes": {
                "penalty": "@xboard_nodepen", "perception_factor": 1}, 
            }
        # add in the correct value of time parameter
        for level in new_journey_levels:
            if level["boarding_cost"]:
                level["boarding_cost"]["on_segments"]["perception_factor"] = fare_perception

    elif class_name == "KNR_TRN_WLK":
        new_journey_levels = copy.deepcopy(journey_levels)

        for i in range(0,len(new_journey_levels)):
            jls = new_journey_levels[i]
            for level in jls["transition_rules"]:
                level["next_journey_level"] = level["next_journey_level"]+1
            jls["transition_rules"].extend(
                [
                {'mode': 'e', 'next_journey_level': i+2},
                {'mode': 'D', 'next_journey_level': len(new_journey_levels)+2}, 
                {'mode': 'w', 'next_journey_level': i+2},
                {'mode': 'p', 'next_journey_level': len(new_journey_levels)+2},
                {'mode': 'k', 'next_journey_level': len(new_journey_levels)+2}
                ]
            )
        
        # level 0: drive access
        transition_rules_drive_access = copy.deepcopy(journey_levels[0]["transition_rules"])
        for level in transition_rules_drive_access:
            level["next_journey_level"] = len(new_journey_levels)+2
        transition_rules_drive_access.extend(
            [
            {'mode': 'e', 'next_journey_level': len(new_journey_levels)+2},
            {'mode': 'D', 'next_journey_level': 0},
            {'mode': 'w', 'next_journey_level': len(new_journey_levels)+2},
            {'mode': 'p', 'next_journey_level': len(new_journey_levels)+2},
            {'mode': 'k', 'next_journey_level': 1}
            ]
        )

        # level 1: use transit
        transition_rules_knr = copy.deepcopy(journey_levels[0]["transition_rules"])
        for level in transition_rules_knr:
            level["next_journey_level"] = 2
        transition_rules_knr.extend(
            [
            {'mode': 'e', 'next_journey_level': len(new_journey_levels)+2},
            {'mode': 'D', 'next_journey_level': len(new_journey_levels)+2},
            {'mode': 'w', 'next_journey_level': len(new_journey_levels)+2},
            {'mode': 'p', 'next_journey_level': len(new_journey_levels)+2},
            {'mode': 'k', 'next_journey_level': 1}
            ]
        )

        # level len(new_journey_levels)+2: every mode is prohibited
        transition_rules_prohibit = copy.deepcopy(journey_levels[0]["transition_rules"])
        for level in transition_rules_prohibit:
            level["next_journey_level"] = len(new_journey_levels)+2
        transition_rules_prohibit.extend(
            [
            {'mode': 'e', 'next_journey_level': len(new_journey_levels)+2},
            {'mode': 'D', 'next_journey_level': len(new_journey_levels)+2},
            {'mode': 'w', 'next_journey_level': len(new_journey_levels)+2},
            {'mode': 'p', 'next_journey_level': len(new_journey_levels)+2},
            {'mode': 'k', 'next_journey_level': len(new_journey_levels)+2}
            ]
        )

        new_journey_levels.insert(
                                0,
                                {
                                "description": "drive access",
                                "destinations_reachable": False,
                                "transition_rules": transition_rules_drive_access,
                                "waiting_time": None,
                                "boarding_time": None,
                                "boarding_cost": None                                     
                                }
        )
        new_journey_levels.insert(
                                1,
                                {
                                "description": "knr",
                                "destinations_reachable": False,
                                "transition_rules": transition_rules_knr,
                                "waiting_time": None,
                                "boarding_time": None,
                                "boarding_cost": None                                     
                                }
        )
        new_journey_levels.append(
                                {
                                "description": "prohibit",
                                "destinations_reachable": False,
                                "transition_rules": transition_rules_prohibit,
                                "waiting_time": None,
                                "boarding_time": None,
                                "boarding_cost": None                                     
                                }
        )

        for level in new_journey_levels[2:-1]:
            level["waiting_time"] = {
                "headway_fraction": "@hdw_fraction",
                "effective_headways": params["effective_headway_source"],
                "spread_factor": 1,
                "perception_factor": "@wait_pfactor"
            }
            level["boarding_time"] = {
            "on_lines": {
                "penalty": "@xboard_penalty", "perception_factor": 1},
            "at_nodes": {
                "penalty": "@xboard_nodepen", "perception_factor": 1}, 
            }
        # add in the correct value of time parameter
        for level in new_journey_levels:
            if level["boarding_cost"]:
                level["boarding_cost"]["on_segments"]["perception_factor"] = fare_perception

    elif class_name == "WLK_TRN_KNR":
        new_journey_levels = copy.deepcopy(journey_levels)

        for i in range(0,len(new_journey_levels)):
            jls = new_journey_levels[i]    
            jls["destinations_reachable"] = False
            jls["transition_rules"].extend(
                [
                {'mode': 'a', 'next_journey_level': len(new_journey_levels)+2},
                {'mode': 'D', 'next_journey_level': len(new_journey_levels)+2}, 
                {'mode': 'w', 'next_journey_level': i+1},
                {'mode': 'p', 'next_journey_level': len(new_journey_levels)+2},
                {'mode': 'k', 'next_journey_level': len(new_journey_levels)+1}
                ]
            )

        # level 0: walk access
        transition_rules_walk_access = copy.deepcopy(journey_levels[0]["transition_rules"])
        for level in transition_rules_walk_access:
            level["next_journey_level"] = 1
        transition_rules_walk_access.extend(
            [
            {'mode': 'a', 'next_journey_level': 0},
            {'mode': 'D', 'next_journey_level': len(new_journey_levels)+2},
            {'mode': 'w', 'next_journey_level': len(new_journey_levels)+2},
            {'mode': 'p', 'next_journey_level': len(new_journey_levels)+2},
            {'mode': 'k', 'next_journey_level': len(new_journey_levels)+2}
            ]
        )    

        # level len(new_journey_levels)+1: drive home
        transition_rules_drive_home = copy.deepcopy(journey_levels[0]["transition_rules"])
        for level in transition_rules_drive_home:
            level["next_journey_level"] = len(new_journey_levels)+2
        transition_rules_drive_home.extend(
            [
            {'mode': 'a', 'next_journey_level': len(new_journey_levels)+2},
            {'mode': 'D', 'next_journey_level': len(new_journey_levels)+1},
            {'mode': 'w', 'next_journey_level': len(new_journey_levels)+2},
            {'mode': 'p', 'next_journey_level': len(new_journey_levels)+2},
            {'mode': 'k', 'next_journey_level': len(new_journey_levels)+2}
            ]
        )

        # level len(new_journey_levels)+2: every mode is prohibited
        transition_rules_prohibit = copy.deepcopy(journey_levels[0]["transition_rules"])
        for level in transition_rules_prohibit:
            level["next_journey_level"] = len(new_journey_levels)+2
        transition_rules_prohibit.extend(
            [
            {'mode': 'a', 'next_journey_level': len(new_journey_levels)+2},
            {'mode': 'D', 'next_journey_level': len(new_journey_levels)+2},
            {'mode': 'w', 'next_journey_level': len(new_journey_levels)+2},
            {'mode': 'p', 'next_journey_level': len(new_journey_levels)+2},
            {'mode': 'k', 'next_journey_level': len(new_journey_levels)+2}
            ]
        )

        new_journey_levels.insert(
                                0,
                                {
                                "description": "walk access",
                                "destinations_reachable": True,
                                "transition_rules": transition_rules_walk_access,
                                "waiting_time": None,
                                "boarding_time": None,
                                "boarding_cost": None                                     
                                }
        )
        new_journey_levels.append(
                                {
                                "description": "drive home",
                                "destinations_reachable": True,
                                "transition_rules": transition_rules_drive_home,
                                "waiting_time": None,
                                "boarding_time": None,
                                "boarding_cost": None                                     
                                }
        )
        new_journey_levels.append(
                                {
                                "description": "prohibit",
                                "destinations_reachable": False,
                                "transition_rules": transition_rules_prohibit,
                                "waiting_time": None,
                                "boarding_time": None,
                                "boarding_cost": None                                     
                                }
        )

        for level in new_journey_levels[1:-2]:
            level["waiting_time"] = {
                "headway_fraction": "@hdw_fraction",
                "effective_headways": params["effective_headway_source"],
                "spread_factor": 1,
                "perception_factor": "@wait_pfactor"
            }
            level["boarding_time"] = {
            "on_lines": {
                "penalty": "@xboard_penalty", "perception_factor": 1},
            "at_nodes": {
                "penalty": "@xboard_nodepen", "perception_factor": 1}, 
            }
        # add in the correct value of time parameter
        for level in new_journey_levels:
            if level["boarding_cost"]:
                level["boarding_cost"]["on_segments"]["perception_factor"] = fare_perception

    elif class_name == "WLK_TRN_WLK":
        new_journey_levels = copy.deepcopy(journey_levels)
        transition_rules = copy.deepcopy(journey_levels[0]["transition_rules"])
        new_journey_levels.insert(
                                0,
                                {
                                "description": "base",
                                "destinations_reachable": True,
                                "transition_rules": transition_rules,
                                "waiting_time": None,
                                "boarding_time": None,
                                "boarding_cost": None                                     
                                }
        )

        for level in new_journey_levels[1:]:
            level["waiting_time"] = {
                "headway_fraction": "@hdw_fraction",
                "effective_headways": params["effective_headway_source"],
                "spread_factor": 1,
                "perception_factor": "@wait_pfactor"
            }
            level["boarding_time"] = {
            "on_lines": {
                "penalty": "@xboard_penalty", "perception_factor": 1},
            "at_nodes": {
                "penalty": "@xboard_nodepen", "perception_factor": 1}, 
            }
        # add in the correct value of time parameter
        for level in new_journey_levels:
            if level["boarding_cost"]:
                level["boarding_cost"]["on_segments"]["perception_factor"] = fare_perception


    with open(
            os.path.join(
                project_dir, "Specifications", "%s_%s_journey_levels.ems" % (period.name, class_name)
            ),
            "w",
    ) as jl_spec_file:
        spec = {"type": "EXTENDED_TRANSIT_ASSIGNMENT", "journey_levels": new_journey_levels}
        _json.dump(spec, jl_spec_file, indent=4)


    return new_journey_levels