import openmatrix as omx
import numpy as np
import pandas as pd
import warnings
from shapely.errors import ShapelyDeprecationWarning
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning) 
import shutil, glob, os
from math import radians, cos, sin, asin, sqrt


class HighwayPostprocessor():
    def __init__(self, input_dir, output_dir):
        """Constructor for the HighwayPostprocessor.

        Args:
            input_dir: directory for input skims, nodes.gpkg
            output_dir: directory for updated skims
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        self.time_periods = ["EA", "AM", "MD", "PM", "EV"]
        self.disconnected_zones = None
        self.disconnected_zones_list = None
        self.disconnected_from_zone_list = None
        self.disconnected_to_zone_list = None
        
    
    def haversine(self, lon1, lat1, lon2, lat2):
        """
        Calculate the great circle distance in kilometers between two points 
        on the earth (specified in decimal degrees)
        """
        # convert decimal degrees to radians 
        lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

        # haversine formula 
        dlon = lon2 - lon1 
        dlat = lat2 - lat1 
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * asin(sqrt(a)) 
        r = 6371 # Radius of earth in kilometers. Use 3956 for miles. Determines return value units.
        return c * r
    
    def identify_disconnected_zones(self):
        # ### Identify disconnected zones (where row or column sum of the distance skim matrix = 0)

        # get taz centroid nodes from network nodes
        #taz_centroids = gpd.read_file(f"{self.input_dir}/nodes.gpkg")
        #taz_centroids = taz_centroids.loc[taz_centroids["model_node_id"] <= 3631, ["model_node_id", "geometry"]].reset_index(drop=True)
        taz_centroids = pd.read_csv(f"{self.input_dir}/taz_centroids.csv")

        # calculate row & column sum of distance skim (using one of the highway skim omx file)
        raw_omx = omx.open_file(f"{self.input_dir}/hwyskmAM.omx")
        dist_skim = np.array(raw_omx["DISTDA"]).astype(np.float32)
        #dist_skim = np.array(raw_omx["TIMEDA"]).astype(np.float32) # was distda
        dist_skim = np.where(dist_skim==np.inf, 0, dist_skim)
        rowsum = dist_skim.sum(axis=1)
        colsum = dist_skim.sum(axis=0)
        #rowsum = list(np.concatenate(dist_skim.sum(axis=1)).flat)
        #colsum = list(np.concatenate(dist_skim.sum(axis=0)).flat)
        raw_omx.close()

        # identify disconnected zones
        zones = taz_centroids.copy()
        zones["skim_rowsum"] = rowsum
        zones["skim_colsum"] = colsum
        disconnected_from_zones = zones[zones["skim_rowsum"] == 0].reset_index(drop=True)
        disconnected_to_zones = zones[zones["skim_colsum"] == 0].reset_index(drop=True)

        # create disconnected zone lists
        self.disconnected_from_zone_list = disconnected_from_zones["model_node_id"].tolist()
        self.disconnected_to_zone_list = disconnected_to_zones["model_node_id"].tolist()
        self.disconnected_zone_list = list(set(self.disconnected_from_zone_list + self.disconnected_to_zone_list))

        # for each zone, find nearest "connected" zone
        nearest_ids = []
        
        def find_dist(row, x, y):
            return self.haversine(row.x, row.y, x, y)
        
        def find_nearest_node(row):
            taz_copy = taz_centroids[(taz_centroids.model_node_id != row.model_node_id) & 
                                    (~taz_centroids.model_node_id.isin(self.disconnected_zone_list)) & 
                                     (taz_centroids.x - row.x).between(-0.5, 0.5)] # additional constraint to speed things up
            taz_copy['dist'] = taz_copy.apply(find_dist, axis = 1, args = (row.x, row.y))
            return taz_copy['model_node_id'][taz_copy['dist'].idxmin()]

        taz_centroids["nearest_id"] = taz_centroids.apply(find_nearest_node, axis = 1)

        # create disconnected zones df (include their corresponding nearest zone id)
        self.disconnected_zones = taz_centroids[taz_centroids["model_node_id"].isin(self.disconnected_zone_list)].reset_index(drop=True)

    # ### Update skim values
    # - borrow skim values from the nearest zone for disconnected zones
    # - fill in intrazonal values (for dist & time skims)

    def update_disconnected_skim_values(self,
        skim_mat: np.matrix,
        disconnected_zones: pd.DataFrame,
        disconnected_from_zone_list: list,
        disconnected_to_zone_list: list,
    ) -> np.matrix:
        updated_skim = skim_mat

        for i in range(len(disconnected_zones)):
            zone_id = disconnected_zones["model_node_id"][i] # index starts from 0
            nearest_id = disconnected_zones["nearest_id"][i]

            if zone_id in disconnected_from_zone_list:
                updated_skim[zone_id - 1] = updated_skim[nearest_id - 1] # replace row values
            if zone_id in disconnected_to_zone_list:
                updated_skim[:, zone_id - 1] = updated_skim[:, nearest_id - 1] # replace col values

        return updated_skim


    def fill_intrazonal_skim_values(self,
        skim_mat: np.matrix,
    ) -> np.matrix:
        updated_skim = skim_mat
        
        
        
        # for each row, find minimum non-zero skim value
        for i in range(updated_skim.shape[0]):
            row_skim_values = updated_skim[i].tolist()[0]
            row_nonzero_values = [x for x in row_skim_values if x != 0]
            if len(row_nonzero_values):
                min_row_skim_value = min(row_nonzero_values)
                updated_skim[i, i] = 0.5 * min_row_skim_value # replace intra-zonal value with min_value
            else:
                updated_skim[i, i] = 0.5 # replace with a small number
        return updated_skim


    def update_skim_values(self):
        
        self.identify_disconnected_zones()
        
        # update skim values
        for period in self.time_periods:
            print(f"period: {period}")
            if os.path.samefile(self.input_dir, self.output_dir):
                if not os.path.exists(os.path.join(self.output_dir, 'temp')):
                    os.mkdir(os.path.join(self.output_dir, 'temp'))
                orig_omx = omx.open_file(f"{self.input_dir}/hwyskm{period}.omx")
                update_omx = omx.open_file(f"{os.path.join(self.output_dir, 'temp')}/hwyskm{period}.omx", "w")
                
            else:
                orig_omx = omx.open_file(f"{self.input_dir}/hwyskm{period}.omx")
                update_omx = omx.open_file(f"{self.output_dir}/hwyskm{period}.omx", "w")
            
            skim_names = orig_omx.list_matrices()

            for skim_name in skim_names:
                print(f"    process {skim_name}")
                skim = np.matrix(orig_omx[skim_name], dtype=np.float32)
                #if skim_name.startswith("DIST") or skim_name.startswith("TIME"):
                updated_skim = np.matrix(np.where(skim==np.inf, 0, skim))
                updated_skim = np.matrix(np.where(updated_skim>=1e6, 0, updated_skim))
                # update disconnected skim values
                
                updated_skim = self.update_disconnected_skim_values(skim_mat=skim,
                                                               disconnected_zones=self.disconnected_zones,
                                                               disconnected_from_zone_list=self.disconnected_from_zone_list,
                                                               disconnected_to_zone_list=self.disconnected_to_zone_list)

                # update intrazonal skim values for distance & time skims
                if skim_name.startswith("DIST") or skim_name.startswith("TIME"):
                    updated_skim = self.fill_intrazonal_skim_values(skim_mat=updated_skim)
                
                if skim_name.startswith("TIME"):
                    updated_skim = np.matrix(np.where(updated_skim>=1e6, 0, updated_skim))
                
                update_omx[skim_name] = updated_skim

            orig_omx.close()
            update_omx.close()
            
        if os.path.samefile(self.input_dir, self.output_dir):
            for src_fn, dst_fn in zip(
                glob.glob(os.path.join(self.output_dir, 'temp','hwyskm*.omx')),
                glob.glob(os.path.join(self.input_dir ,'hwyskm*.omx'))):
                
                os.remove(dst_fn)
                shutil.copy(src_fn, dst_fn)
                os.remove(src_fn)


class TransitPostprocessor():
    def __init__(self, input_dir, output_dir):
        """Constructor for the TransitPostprocessor.

        Args:
            input_dir: directory for input skims
            output_dir: directory for updated skims
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        self.time_periods = ["EA", "AM", "MD", "PM", "EV"]
        self.attr_lookup = {
            'FIRSTWAIT': 'IWAIT',
            'TOTALWAIT': 'WAIT',
            'XFERS': 'BOARDS',
            'DTIME': 'DTIME',
            'DDIST': 'DDIST',
            'WACC': 'WACC',
            'WEGR': 'WEGR',
            'LBIVTT': 'IVTLOC',
            'EBIVTT': 'IVTEXP',
            'LRIVTT': 'IVTLRT',
            'HRIVTT': 'IVTHVY',
            'CRIVTT': 'IVTCOM',
            'FRIVTT': 'IVTFRY',
            'LBPIVTT': 'PIVTLOC',
            'EBPIVTT': 'PIVTEXP',
            'LRPIVTT': 'PIVTLRT',
            'HRPIVTT': 'PIVTHVY',
            'CRPIVTT': 'PIVTCOM',
            'FRPIVTT': 'PIVTFRY',
            'XFERWAIT': 'XWAIT',
            'FARE': 'FARE',
            'WAUX': 'WAUX',
            'TOTALIVTT': 'IVT',
        }
    def update_skim_names(self):

        if os.path.samefile(self.input_dir, self.output_dir):
            if not os.path.exists(os.path.join(self.output_dir, 'temp')):
                os.mkdir(os.path.join(self.output_dir, 'temp'))
        
        skim_filenames = [os.path.basename(fn) for fn in glob.glob(
            os.path.join(self.input_dir, 'trn*.omx'))]
        for skim_fn in skim_filenames:
            orig_omx = omx.open_file(os.path.join(self.input_dir, skim_fn))
            if os.path.samefile(self.input_dir, self.output_dir):
                update_omx = omx.open_file(os.path.join(self.output_dir, 'temp',skim_fn), "w")
            else:
                update_omx =  omx.open_file(os.path.join(self.output_dir ,skim_fn), "w")
        
            skim_names = orig_omx.list_matrices()

            for skim_name in skim_names:
                skim = np.matrix(orig_omx[skim_name], dtype=np.float32)
                if skim_name in self.attr_lookup:
                    new_skim_name = self.attr_lookup[skim_name]
                elif skim_name in self.attr_lookup.values():
                    new_skim_name = skim_name
                else:
                    new_skim_name = skim_name 
                update_omx[new_skim_name] = skim

            orig_omx.close()
            update_omx.close()
            
        if os.path.samefile(self.input_dir, self.output_dir):
            for src_fn, dst_fn in zip(
                glob.glob(os.path.join(self.output_dir, 'temp','trn*.omx')),
                glob.glob(os.path.join(self.input_dir ,'trn*.omx'))):
                
                os.remove(dst_fn)
                shutil.copy(src_fn, dst_fn)
                os.remove(src_fn)
