import os
import pandas as pd
import numpy as np
import openmatrix as omx
from tm2py.components.component import Component
from tm2py.components.demand.prepare_demand import PrepareHighwayDemand
from tm2py.emme.matrix import OMXManager
import string


class ConvergenceReport(Component):
    """Produce Highway and Transit assignment convergence metrics.

    Args:
        controller: parent RunController object
    """

    def __init__(self, controller):
        """Constructor for ConvergenceReport components.

        Args:
            controller (RunController): Reference to current run controller.
        """
        super().__init__(controller)
        self.config =  self.controller.config

    def run(self):
        #self.save_results()
        if self.controller.iteration > self.controller.config.run.start_iteration:
            self.export_summaries()
        
        
    def validate_inputs(self):
        pass
        
    def _get_link_attributes(self, network, attributes):
        """Calculate the length attributes used in the highway skims."""
        link_attr_table = []
        for link in network.links():
            link_attr_table.append({
                attr:link[attr] for attr in attributes
            })
        return pd.DataFrame(link_attr_table)
    
    def export_trip_tables(self):
        prep_demand = PrepareHighwayDemand(self.controller)
        
        # Trip tables: total daily trips by mode/vehicle class in matrices
        prep_demand.export_trip_tables()
        
    def export_skims(self):
    
        num_internal_zones = self.num_internal_zones
        
        # Skims: AM / PM WAT IVT and DA time
        skims = omx.open_file(self.get_abs_path(self.controller.config.highway.convergence.output_skim_path.format(iteration = self.controller.iteration)),
            "w")
        
        # for time_period in self.controller.config.highway.convergence.skim_selected_time_periods:
        for time_period in ['am','pm']:
            trn_skm_fn = self.get_abs_path(self.controller.config.transit.output_skim_path / 
                self.controller.config.transit.output_skim_filename_tmpl.format(time_period = time_period, set_name = 'wlk_trn_wlk'))
            with omx.open_file(trn_skm_fn) as trn_skm:
                ivt = np.array(trn_skm['IVT'])[:num_internal_zones, :num_internal_zones]
            skims[f'TRN_IVT_{time_period}'] = ivt
            
            hwy_skm_fn = self.get_abs_path(self.controller.config.highway.output_skim_path / 
                self.controller.config.highway.output_skim_filename_tmpl.format(time_period = time_period))
            with omx.open_file(hwy_skm_fn) as hwy_skm:
                timeda = np.array(hwy_skm['TIMEDA'])[:num_internal_zones, :num_internal_zones]
            skims[f'HWY_DA_TIME_{time_period}'] = timeda
        
        skims.close()
        
    def export_network_attrs(self):
    
        # Network attributes
        network_attrs = ['#link_id','length','auto_time','auto_volume']
        summary_by_time_period = {}
        
        for time in self.time_period_names:
            scenario = self.get_emme_scenario(
                self.config.emme.highway_database_path, time)
            network = scenario.get_network()
            attrs = network_attrs
            link_attrs = self._get_link_attributes(network, attrs)
            link_attrs['vmt'] = link_attrs['auto_volume'] * link_attrs['length']
            link_attrs['tt'] = link_attrs['auto_volume'] * link_attrs['auto_time']
            if time.upper() == 'AM':
                network_attr_fn = self.get_abs_path(
                    self.controller.config.highway.convergence.output_network_attr_filename.format(iteration = self.controller.iteration))
                link_attrs.set_index('#link_id').to_parquet(network_attr_fn)
            summary = link_attrs[['auto_volume','vmt','tt']].sum()
            summary_by_time_period[time] = summary
        
        summary_all = pd.concat(summary_by_time_period, axis = 1).T
        summary_all.index.rename('time_period', inplace = True)
        network_summary_fn = self.get_abs_path(
            self.controller.config.highway.convergence.output_network_summary_filename.format(iteration = self.controller.iteration))
        summary_all.to_csv(network_summary_fn)
    
    
    def save_results(self):
        """Save out key assignment results that are used for summaries in the next iteration."""
        self.export_trip_tables()
        self.export_skims()
        self.export_network_attrs()
        
    def generate_trip_table_stats(self, lines_out):
        
        lines_out.append('------Trip Table Convergence Statistics------')
        trip_tables_prev_iter = self.get_abs_path(self.controller.config.highway.convergence.output_triptable_path.format(iteration = self.controller.iteration - 1))
        trip_tables_curr_iter = self.get_abs_path(self.controller.config.highway.convergence.output_triptable_path.format(iteration = self.controller.iteration))
        
        curr_iter = omx.open_file(trip_tables_curr_iter)
        prev_iter = omx.open_file(trip_tables_prev_iter)
        
        trip_classes = curr_iter.list_matrices()
        trip_totals = []
        MSE = 0

        for cl in trip_classes:
            curr_arr =  np.array(curr_iter[cl])
            prev_arr =  np.array(prev_iter[cl])
            trip_totals.append([cl, curr_arr.sum(), prev_arr.sum()])
            MSE += np.sqrt(((curr_arr - prev_arr) ** 2).sum())
        prev_iter.close()
        curr_iter.close()
        
        trip_totals = pd.DataFrame(trip_totals, columns = ['Mode Class', 'Trips', 'PrevIter'])
        trip_totals['Diff'] = trip_totals['Trips'] - trip_totals['PrevIter']
        
        lines_out.append(trip_totals.to_csv(index = False, sep = '\t'))
        
        lines_out.append(f'Difference in total daily trips = \t{trip_totals["Diff"].sum()}')
        lines_out.append(f'Percent RMSE in total daily trips = \t{MSE / trip_totals["PrevIter"].sum()}')

    def generate_skim_stats(self, lines_out):
        
        lines_out.append('------Skim Convergence Statistics------')
        skims_prev_iter = self.get_abs_path(self.controller.config.highway.convergence.output_skim_path.format(iteration = self.controller.iteration - 1))
        skims_curr_iter = self.get_abs_path(self.controller.config.highway.convergence.output_skim_path.format(iteration = self.controller.iteration))
        
        curr_iter = omx.open_file(skims_prev_iter)
        prev_iter = omx.open_file(skims_curr_iter)
        
        for mode in ['TRN_IVT', 'HWY_DA_TIME']:
            # for time_period in self.controller.config.highway.convergence.skim_selected_time_periods:
            for time_period in ['am','pm']:
                curr_arr =  np.array(curr_iter[f'{mode}_{time_period}'])
                prev_arr =  np.array(prev_iter[f'{mode}_{time_period}'])
                RMSPE = np.sqrt(((curr_arr - prev_arr) ** 2).sum()) / prev_arr.sum()
                
                lines_out.append(f'Total skim travel time in {time_period} {mode}: \t{curr_arr.sum()} \t Difference: \t {curr_arr.sum() - prev_arr.sum()}')
                lines_out.append(f'Percent RMSE in {time_period} {mode}: \t{RMSPE}')
                
        prev_iter.close()
        curr_iter.close()
        
    def generate_network_stats(self, lines_out):
    
        lines_out.append('------Network-based Convergence Statistics------')
        network_attr_prev_iter = pd.read_parquet(
            self.get_abs_path(self.controller.config.highway.convergence.output_network_attr_filename.format(
            iteration = self.controller.iteration - 1)))
            
        network_attr_curr_iter =  pd.read_parquet(
            self.get_abs_path(self.controller.config.highway.convergence.output_network_attr_filename.format(
            iteration = self.controller.iteration)))
            
        network_summary_prev_iter = pd.read_csv(
            self.get_abs_path(self.controller.config.highway.convergence.output_network_summary_filename.format(
            iteration = self.controller.iteration - 1)))
            
        network_summary_curr_iter = pd.read_csv(
            self.get_abs_path(self.controller.config.highway.convergence.output_network_summary_filename.format(
            iteration = self.controller.iteration)))
        
        # number of links with change in volumes exceeding 5%
        link_vols = pd.concat([network_attr_prev_iter['auto_volume'], network_attr_curr_iter['auto_volume']],
            axis = 1)
        link_vols.columns =  ['prev', 'curr']
        link_vols['diff'] = link_vols['curr'] - link_vols['prev']
        link_vols['pct_diff'] = link_vols['diff'] / link_vols['prev'].clip(lower = 0.1)
        lines_out.append(f'Number of links with AM volume change greater than 5%: \t {(link_vols.pct_diff > 0.05).sum()}, ({100*(link_vols.pct_diff > 0.05).mean()}%)')
        
        # Network VMT, AM and daily
        AM_VMT_prev, AM_VMT_curr = network_summary_prev_iter[network_summary_prev_iter['time_period'] == "am"]['vmt'].sum(), network_summary_curr_iter[network_summary_curr_iter['time_period'] == "am"]['vmt'].sum()
        daily_VMT_prev, daily_VMT_curr = network_summary_prev_iter['vmt'].sum(), network_summary_curr_iter['vmt'].sum()
        lines_out.append(f'AM highway network VMT:\t {AM_VMT_curr}')
        lines_out.append(f'Change in AM highway network VMT: \t {(AM_VMT_curr - AM_VMT_prev) / AM_VMT_prev}')
        lines_out.append(f'Daily highway network VMT:\t {daily_VMT_curr}')
        lines_out.append(f'Change in daily highway network VMT: \t {(daily_VMT_curr - daily_VMT_prev) / daily_VMT_prev}')
        
        # Network travel time, AM and daily
        AM_TT_prev, AM_TT_curr = network_summary_prev_iter[network_summary_prev_iter['time_period'] == "am"]['tt'].sum(), network_summary_curr_iter[network_summary_curr_iter['time_period'] == "am"]['tt'].sum()
        daily_TT_prev, daily_TT_curr = network_summary_prev_iter['tt'].sum(), network_summary_curr_iter['tt'].sum()
        lines_out.append(f'AM highway network travel time:\t {AM_TT_curr}')
        lines_out.append(f'Change in AM highway network travel time: \t {(AM_TT_curr - AM_TT_prev) / AM_TT_prev}')
        lines_out.append(f'Daily highway network travel time:\t {daily_TT_curr}')
        lines_out.append(f'Change in daily highway network travel time: \t {(daily_TT_curr - daily_TT_prev) / daily_TT_prev}')
        
    def export_summaries(self):
            
        iteration = self.controller.iteration
        report_path = self.get_abs_path(self.config.highway.convergence.output_convergence_report_path)
                
        lines_out = []
        lines_out.append(f'========== Iter {self.controller.iteration} ==========')
        lines_out.append('')
        
        self.generate_trip_table_stats(lines_out)

        self.generate_skim_stats(lines_out)
        
        self.generate_network_stats(lines_out)
        
        lines_out.append('')

        report_f = open(report_path, 'a')
        report_f.writelines([l + '\n' for l in lines_out])
        
        report_f.close()
    
    
    @property
    def num_internal_zones(self):
        return len(pd.read_csv(
            self.get_abs_path(self.controller.config.scenario.landuse_file), usecols = [self.controller.config.scenario.landuse_index_column]))
        