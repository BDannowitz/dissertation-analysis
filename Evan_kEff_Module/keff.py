import numpy as np
import pandas as pd
import uncertainties as unc
from uncertainties import unumpy as unp
from itertools import product
import pymysql as Mdb
from scipy.optimize import curve_fit


def lin_func(x, b, m):
    """
    A linear function:
    
    f(x) = b + m*x
    """
    return b + np.multiply(m,x)


#def exp_func(x, a, c):
#    """
#    An exponential decay function:
#    f(x) = a*exp(-c*x)
#    """
#    if isinstance(a,unc.UFloat) or isinstance(c,unc.UFloat):
#        return a * unp.exp(-c*x)
#    else:    
#        return a * np.exp(-c*x)


def exp_func(x, c):
    """
    An exponential decay function:
    f(x) = exp(-c*x)
    """
    if isinstance(c,unc.UFloat):
        return unp.exp(-c*x)
    else:    
        return np.exp(-c*x)


def w_score_interval(p, x, kind='upper'):
    base = (p + 0.5*x)/(1.0+x)
    uncertainty = np.sqrt(p*x*(1.0-p) + 0.25*np.square(x))/(1.0 + x)
    if kind=='upper':
        return base + uncertainty
    elif kind=='lower':
        return base - uncertainty


class kEffCorrection(object):
    """An object for handling the kEfficiency correction."""

    def __init__(self, roadset, targetPos, mass_range=(4.2,10.0),
                 kin=['xT'],
                 kin_bins=[[0.08, 0.14, 0.16, 0.18, 0.21, 0.25, 0.31, 0.53]],
                 intensity=[
                     0, 10000, 20000, 30000,
                     40000, 50000, 60000, 70000,
                     80000, 90000, 100000
                 ],
                 schema='user_recon_rate_dep',
                 merged_version='R005_V001',
                 server='seaquel.physics.illinois.edu'):
        """Get Messy & Clean data, calculate efficiencies, and fit some
        functions to them.

        Keyword arguments:
        roadset -- identifying number of the roadset to use (57, 62, 67)
        targetPos -- position identifier of the target (1,3,5,6,7)
        mass_range -- range of dimuon masses to analyze (default: (4.2, 10.0))
        kin -- a *list* of kinematics by which to bin the messy and clean data
        kin_bins -- a *list* of binnings for each respective kinematic in the
                    "kin" argument. If a scalar, the data is broken up into
                    that number of statistically equan bins. If a list of
                    values, they will be used as the bin edges.
        schema -- the MySQL database schema where the messy and clean tables
                  can be found (default: "user_recon_rate_dep")
        merged_version -- the suffix of the messy/clean tables use a certain
                          merged production. (default: "R005_V001")
        server -- The MySQL server where one can find the schema
        """
        
        # Create containers
        self.clean_df = pd.DataFrame()
        self.messy_df = pd.DataFrame()
        self.eff_df = pd.DataFrame()
        self.lin_df = pd.DataFrame()
        self.exp_df = pd.DataFrame()

        # Store relevant criteria
        self.intensity = intensity
        self.intensity_centers = (np.add(intensity[1:],intensity[:-1]))/2
        self.targetPos = targetPos
        self.roadset = roadset
        self.mass_range = mass_range
        self.kin = kin
        self.kin_bins = kin_bins
        self.merged_version = merged_version
        
        # Manifest queries
        self.clean_query = self.make_query(kind='clean')
        self.messy_query = self.make_query(kind='messy')
        
        # Fetch relevant dimuon data
        self.clean_df = self.get_dimuons(self.clean_query, server, schema)
        self.messy_df = self.get_dimuons(self.messy_query, server, schema)
       
        # Clean out messy dimuons where there are no clean dimuons
        self.messy_df = self.require_clean(self.messy_df, self.clean_df)

        # Get efficiencies for each kinematic and intensity bin
        self.eff_df = self.calc_eff()
        
        # Calculate linear fits to efficiency as a function of intensity
        #self.lin_df = self.get_fits(func=lin_func)
        self.exp_df = self.get_fits(func=exp_func)
        
        del self.clean_df
        del self.messy_df
        #del self.eff_df

        return None
   
    def get_rf_cut(self):
        # RF < BeamDAQ.Inh_thres for RF's -8 to 8
        #rf_str = "`RF+00` < b.Inh_thres AND "
        #for i in range(1,9):
        #    rf_str += ("`RF+%.02d` < b.Inh_thres AND " % i)
        #    rf_str += ("`RF-%.02d` < b.Inh_thres AND " % i)
        
        # RF > 0 for RF's -16 to 16
        rf_str = " AND `RF+00` > 0 AND "
        for i in range(1,17):
            rf_str += ("`RF+%.02d` > 0 AND " % i)
            rf_str += ("`RF-%.02d` > 0 AND " % i)
        
        # Chop off the trailing ' AND '
        rf_str = rf_str[:-5]
        
        return rf_str
    
    
    def make_query(self, kind='clean'):
        
        kin_str = ''
    
        # Select desired intensity range
        where = ("WHERE chamber_intensity >= %d AND chamber_intensity < %d" %
                 (self.intensity[0], self.intensity[-1]))
        # Add mass range requirement, if needed
        if self.mass_range and len(self.mass_range)==2:
            where += (" AND mass > %f AND mass < %f" %
                      (self.mass_range[0], self.mass_range[1]))
        # Select on cut kinematics and add ranges to WHERE clause
        for kin, bins in zip(self.kin, self.kin_bins):
            if kin=='theta':
                kin_str += 'ACOS(costh) AS `theta`,'
            elif kin == 'dpt':
                kin_str += 'SQRT(dpx*dpx + dpy*dpy) AS `dpt`,'
            else:
                kin_str += kin + ','
            if isinstance(bins, list):
                where += (" HAVING %s >= %f AND %s < %f" %
                          (kin, bins[0], kin, bins[-1]))
        #where+=(" AND dpt >= 0.6")

        #where += self.get_rf_cut()
            
        # Assemble SELECT clause
        #select = ("SELECT dimuonID, runID, eventID, %s chamber_intensity, gmcWeight, "
        #          "POW(gmcWeight,2) AS `gmcWeight_sq` FROM" % kin_str)
        select = ("""SELECT dimuonID, runID, eventID, mass, xT, xB, xF, dpz,
                     costh, phi, ACOS(costh) AS `theta`, SQRT(dpx*dpx + dpy*dpy) AS `dpt`,
                     chamber_intensity, gmcWeight, POW(gmcWeight,2) AS `gmcWeight_sq` FROM""")

        table = ("""%s_%d_%d_%s""" %
                    #INNER JOIN merged_roadset%d_%s.QIE q USING(spillID, eventID)
                    #INNER JOIN merged_roadset%d_%s.BeamDAQ b USING(spillID)""" %
                    (kind, self.roadset, self.targetPos, self.merged_version))#,
                    # self.roadset, self.merged_version,
                    # self.roadset, self.merged_version))
        # Put all together into two queries
        query = ("%s %s %s" % (select, table, where))
        
        return query

    
    def get_dimuons(self, query, server, schema):
        
        port = 3306
        df = pd.DataFrame()
        if server=='seaquel.physics.illinois.edu':
            port = 3283
    
        try:
            db = Mdb.connect(read_default_file='../.my.cnf',
                             read_default_group='guest',
                             db=schema,
                             host=server,
                             port=port)
            df = pd.read_sql(query, db, index_col='dimuonID')
            
            if db:
                db.close()

        except Mdb.Error, e:
            print "Error %d: %s" % (e.args[0], e.args[1])
        
        if 'costh' in df.columns:
            df['theta'] = np.arccos(df.costh)
        if all(item in df.columns for item in ('dpx', 'dpy')):
            df['dpt'] = np.sqrt(np.square(df.dpx)+np.square(df.dpy))

        return df
   

    def require_clean(self, messy_df, clean_df):
        """Remove entries from messy where there is not a dimuon in clean."""
        columns = messy_df.columns
        run_event = clean_df[['runID', 'eventID']]
        messy_df = pd.merge(run_event, messy_df, how='left', on=['runID', 'eventID']).dropna()

        return messy_df
            

    def calc_eff(self, wilson_l=1.0):
        
        eff_df = pd.DataFrame(columns=[
            'raw_clean', 'weighted_clean', 'raw_messy',
            'weighted_messy', 'efficiency', 'uncertainty'
            ])
        
        # Create the cuts that the data will be grouped by
        clean_cuts = []
        messy_cuts = []
        indexes = range(0,len(self.kin)+1)
        # Cut on kinematic bins
        for ix, kin, bins in zip(indexes, self.kin, self.kin_bins):
            if isinstance(bins, list) or isinstance(bins, np.ndarray):
                clean_cuts.append(pd.cut(self.clean_df[kin],
                                         bins, right=True))
                messy_cuts.append(pd.cut(self.messy_df[kin],
                                         bins, right=True))
            elif isinstance(bins, int):
                #clean_cuts.append(pd.qcut(self.clean_df[kin], bins))
                _, bins = pd.qcut(self.clean_df[kin], bins,
                                  retbins=True)
                self.kin_bins[ix] = bins
                messy_cuts.append(pd.cut(
                    self.messy_df[kin], self.kin_bins[ix], right=True))
                clean_cuts.append(pd.cut(
                    self.clean_df[kin], self.kin_bins[ix], right=True))

        # And cut on intensity bins
        clean_cuts.append(pd.cut(self.clean_df['chamber_intensity'],
                                 self.intensity, right=True))
        messy_cuts.append(pd.cut(self.messy_df['chamber_intensity'],
                                 self.intensity, right=True))
     
        # Get raw and weighted yields 
        eff_df['raw_clean'] = (self.clean_df
                                   .groupby(by=clean_cuts)
                                   .gmcWeight
                                   .count())
        eff_df['weighted_clean'] = (self.clean_df
                                        .groupby(by=clean_cuts)
                                        .gmcWeight
                                        .sum())
        eff_df['raw_messy'] = (self.messy_df
                                   .groupby(by=messy_cuts)
                                   .gmcWeight
                                   .count())
        eff_df['weighted_messy'] = (self.messy_df
                                        .groupby(by=messy_cuts)
                                        .gmcWeight
                                        .sum())

        # Calculate the sums of the squares of the weights
        eff_df['ssqw_clean'] = (self.clean_df
                                    .groupby(by=clean_cuts)
                                    .gmcWeight_sq
                                    .sum())
        eff_df['ssqw_messy'] = (self.messy_df
                                    .groupby(by=messy_cuts)
                                    .gmcWeight_sq
                                    .sum())
       
        # Calculate the 
        eff_df['weighted_other'] = (eff_df.weighted_clean - 
                                    eff_df.weighted_messy)
        eff_df['ssqw_other'] = np.abs(eff_df.ssqw_clean - eff_df.ssqw_messy)
       
        # Calculate the efficiencies and uncertainties
        eff_df['raw_efficiency'] = eff_df.raw_messy / eff_df.raw_clean
        eff_df['efficiency'] = eff_df.weighted_messy / eff_df.weighted_clean
        
        # Using the Wilson Score Interval
        #eff_df['wilson_x'] = (
        #        wilson_l**2 * np.divide(eff_df.ssqw_clean,
        #                                np.square(eff_df.weighted_clean)))
        #eff_df['upper_interval'] = w_score_interval(eff_df.efficiency,
        #                                            eff_df.wilson_x, kind='upper')
        #eff_df['lower_interval'] = w_score_interval(eff_df.efficiency,
        #                                            eff_df.wilson_x, kind='lower')
        # Currently cannot fit to asymmetric uncertainties... so just average them.
        #eff_df['uncertainty'] = (((eff_df.upper_interval-eff_df.efficiency) +
        #                          (eff_df.efficiency-eff_df.lower_interval))/2.0)

        # Using regular weighter binomial statistics
        eff_df['uncertainty'] = np.divide(
            np.sqrt(eff_df.ssqw_messy*np.square(eff_df.weighted_other) + 
                    eff_df.ssqw_other*np.square(eff_df.weighted_messy)),
            np.square(eff_df.weighted_clean))
        
        # Calculate weighted intensity 
        #   This is used for calculating intensity weighted means
        self.clean_df['w_intensity'] = (self.clean_df.chamber_intensity * 
                                        self.clean_df.gmcWeight)
        eff_df['mean_intensity'] = (self.clean_df
                                        .groupby(by=clean_cuts)
                                        .chamber_intensity
                                        .mean())
        eff_df['w_mean_intensity'] = (np.divide(self.clean_df
                                                    .groupby(by=clean_cuts)
                                                    .w_intensity
                                                    .sum(),
                                                eff_df.weighted_clean))
        
        return eff_df
   
    
    def get_chisq_pdf(self, x, y, sigma, func, n):
        degree = float(len(x)-1-n)
        chisq = 0.0

        for x_i, y_i, sig_i in zip(x, y, sigma):
            chisq += (y_i - func(x_i))**2/sig_i**2

        return chisq / degree


    def get_fits(self, func=exp_func):
        """
        Internal function to calculate the linear fits to kinematic bins
        for which there is enough data.
        
        
        """
        
        fits_list = []
        local_eff_df = self.eff_df.dropna(how='any').copy()
        
        if func==lin_func:
            p0 = [1.0, -1e-5]
        elif func==exp_func:
            p0 = [1.2e-5] 
        
        if len(self.kin) > 0:
            # Use itertools.product to get every combination
            #   all of the kinematic bins
            for bins in list(product(*self.eff_df.index.levels[:-1])):
                # Create a pandas query to select on that
                #   multi-dimensional bin
                where = ''
                n = 0
                for kin in self.kin:
                    where += ('%s == @bins[%d] and ' % (kin, n))
                    n += 1
                # Chop off trailing ' and '
                where = where[:-5]
                # Collect slice of data
                slice_df = local_eff_df.query(where).copy()

                if len(slice_df)>(0.5*len(self.intensity_centers)):
                    # Fit a linear curve to the efficiency
                    popt, pcov = curve_fit(func,
                                           slice_df.w_mean_intensity,
                                           slice_df.efficiency,
                                           p0=p0,
                                           sigma=slice_df.uncertainty)
                    # Calculate the chi-squared of that fit
                    chisq_pdf = self.get_chisq_pdf(
                            slice_df.w_mean_intensity,
                            slice_df.efficiency,
                            slice_df.uncertainty,
                            lambda x: func(x, *popt),
                            2)
                    # Add parameters, uncertainties, and chi-squareds
                    #   to the list.
                    if len(popt)>1:
                        fits_list.append(
                            [x for x in bins] + 
                            [popt[0], np.absolute(pcov[0,0])**0.5,
                             popt[1], np.absolute(pcov[1,1])**0.5, chisq_pdf]
                        )
                        fits_df = pd.DataFrame(
                            fits_list,
                            columns=(self.eff_df.index.names[:-1] +
                                     ['p0', 'p0_unc', 'p1', 'p1_unc', 'chisq_pdf'])
                        )
                    else:
                        fits_list.append(
                            [x for x in bins] + 
                            [popt[0], np.absolute(pcov[0,0])**0.5, chisq_pdf]
                        )
                        fits_df = pd.DataFrame(
                            fits_list,
                            columns=(self.eff_df.index.names[:-1] +
                                     ['p0', 'p0_unc', 'chisq_pdf'])
                        )
                    
            if len(popt)>1:
                fits_df = pd.DataFrame(
                    fits_list,
                    columns=(self.eff_df.index.names[:-1] + 
                             ['p0', 'p0_unc', 'p1', 'p1_unc', 'chisq_pdf']
                    )
                )
            else:
                fits_df = pd.DataFrame(
                    fits_list,
                    columns=(self.eff_df.index.names[:-1] + 
                             ['p0', 'p0_unc', 'chisq_pdf']
                    )
                )
            fits_df.set_index(self.eff_df.index.names[:-1],
                              drop=True, inplace=True)
        else:
            popt, pcov = curve_fit(func,
                                   local_eff_df.w_mean_intensity,
                                   local_eff_df.efficiency,
                                   p0=p0,
                                   sigma=local_eff_df.uncertainty)
            chisq_pdf = self.get_chisq_pdf(
                local_eff_df.w_mean_intensity,
                local_eff_df.efficiency,
                local_eff_df.uncertainty,
                lambda x: func(x, *popt),
                2)
            if len(popt)>1:
                fits_list.append([popt[0], np.absolute(pcov[0,0])**0.5,
                                  popt[1], np.absolute(pcov[1,1])**0.5,
                                  chisq_pdf])
                fits_df = pd.DataFrame(
                    fits_list,
                    columns=(['p0', 'p0_unc', 'p1', 'p1_unc', 'chisq_pdf'])
                )
            else:
                
                fits_list.append([popt[0], np.absolute(pcov[0,0])**0.5,
                                  chisq_pdf])
                fits_df = pd.DataFrame(
                    fits_list,
                    columns=(['p0', 'p0_unc', 'chisq_pdf'])
                )

        return fits_df


    def kEff(self, x, ret_unc=False, inv=False, func=exp_func):
        """
        Main operating function of this class.
        
        Feed it in dimuon data with the same number of fields per row
          as self.kinematics + intensity and return kEff factors
          
        Example:
          You have created a kEffCorrection object with the kinematics
          binned in xT and xB. In order to use this function, you must
          feed kEff() an array or array of arrays of [xT, xB, intensity],
          in that order.

        Output:
          A weight to apply to that dimuon. If an array is given, an
          array of weights is given.
          
        Options:
          'ret_unc': Boolean. If true, the array will be a tuple of
                     nominal and stddev values
          'inv': Boolean. If true, will return 1/kEff, which may be
                 desirable
        """
        # Verify there is data, and the right number of values per row.
        x = np.array(x)
        if len(x)==0:
            print "Empty set."
            return None
        if x.shape[1] != (len(self.kin) + 1):
            print ("Incorrect number of values per row. "
                   "There should be %d values per row: %s, and intensity" % 
                   (len(self.kin)+1, ', '.join(self.kin)))
            return None
        
        # Convert to DataFrame
        x_df = pd.DataFrame(x, columns=(self.kin + ['intensity']))
        
        # First group these dimuons into the same kinematic binning as
        #   used when calculating the efficiencies.
        cuts = []
        for kin, bins in zip(self.kin, self.kin_bins):
            cuts.append(pd.cut(x_df[kin], bins, right=True))
        x_grp = x_df.groupby(by=cuts)

        # Rename the kinematic columns to make room for the binnin columns
        x_df = x_df.rename(columns={x:x+'_x' for x in self.kin})
        for kin in self.kin:
            x_df[kin] = None

        # Now we have the groups along with the indexes of the DataFrame
        #   that are in each bin. Let's put this all into a single list
        for group in x_grp.groups:
            indexes = x_grp.groups[group]
            x_df.ix[indexes, self.kin] = group

        _ = x_df.set_index(self.kin, inplace=True)
        
        # Get the fit parameters matched up to each row
        if func==lin_func:
            x_df = x_df.merge(self.lin_df, left_index=True, right_index=True,
                              how='left', sort=False)
        elif func==exp_func:
            x_df['p0'] = None
            x_df['p0'] = x_df.apply(lambda x: self.exp_df.ix[x.index]['p0'], axis=0)
            x_df['p0_unc'] = None
            x_df['p0_unc'] = x_df.apply(lambda x: self.exp_df.ix[x.index]['p0_unc'], axis=0)

        # Apply the fit parameters
        if func==lin_func:
            x_df['weight'] = x_df.apply(
                lambda x: func(x.intensity, x.p0, x.p1), axis=1)
            
            x_df['weight_unc'] = x_df.apply(
                lambda x: unc.std_dev(func(x.intensity,
                                           unc.ufloat(x.p0, x.p0_unc), 
                                           unc.ufloat(x.p1, x.p1_unc)
                                          )),
                axis=1)
        elif func==exp_func:
            x_df['weight'] = x_df.apply(
                lambda x: func(x.intensity, x.p0), axis=1)
            
            x_df['weight_unc'] = x_df.apply(
                lambda x: unc.std_dev(func(x.intensity,
                                           unc.ufloat(x.p0, x.p0_unc), 
                                           )
                                      ),
                axis=1)
        
        # Inverse the weights and their uncertainties if requested
        if inv:
            if ret_unc:
                x_df['weight_unc'] = x_df['weight_unc']/(x_df['weight']**2)
            x_df['weight'] = 1.0/x_df['weight']
        
        # Return both weight and uncertainty if requested
        if ret_unc:
            return x_df[['weight', 'weight_unc']].values
        else:
            return x_df['weight'].values

