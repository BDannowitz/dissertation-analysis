#!/home/dannowi1/anaconda2/bin/python
"""
Track Efficiency.

Usage:
    ./Track-Efficiency.py -N <N> [(--set <set> --nsets <nsets>)] [--target <pos>]

Options:
    -N <N>	     Number of times to simulate each (roadID, intensity)
		     [Default: 100]
    --nsets <nsets>  Break the roadset up into N sets [Default: 1]
    --set <set>      If broken up roadset, then work on the Nth
              	     subset [Default: 1]
    --target <pos>   Target position to simulate
"""


import numpy as np
import pandas as pd
import docopt


chamber_eff = pd.DataFrame()
for i in xrange(1,8):
    temp_df = pd.read_csv('data/chamber_efficiency_targ_%d.txt' % i, sep='\t')
    temp_df.insert(0, "target", i)
    chamber_eff = pd.concat([chamber_eff, temp_df])
chamber_eff['midElementID'] = ((chamber_eff['maxElementID'] + chamber_eff['minElementID'])/2.0)
chamber_eff['section'] = np.floor(chamber_eff.midElementID/(chamber_eff.maxElementID - chamber_eff.minElementID))
chamber_eff['section'] = chamber_eff['section'].astype(int)
chamber_eff = chamber_eff.drop('midElementID', axis=1)

# # Procedure
# 
# ### For each roadID in the roadset, determine the efficiency of a track passing through that road.
# 
# 1. Find all the paddles associated with a road
# 2. Find the central wire for each plane that's masked by each hodoscope paddle
# 3. Simulate a track going through that road by throwing a random number against all the efficiencies of all 18 planes
# 4. Test to see if the hits that made it would have constituted a track
# 5. Report these 'roadID' track efficiencies for each intensity bin

# # Functions
# 
# * **get_mean_wires_from_hodo(stationID, detectorHalf, elementID)**: Each paddle corresponds to a range of wireElementID's in each plane it shares a stationID with. Find the middle elementID of that range.
# * **get_mean_wires_from_road(roadID)**: Get each paddle for a given roadID. Call the above function to find the mean wireElementID's for all 18 planes in for the roadID.
# * **get_efficiency(plane, element, intensity)**: For a given plane, for a given wireElementID, for a given intensity, return the efficiency and the uncertainty in that efficiency.
# * **test_track(hits)**: Take a set of detectors and their hits and test against "good track" criteria to see if it passes all criteria.
# * **road_test(roadID, intensity, N)**: For a given roadID and intensity, fetch the efficiencies (and their uncertainties) for all corresponding planes. Then, generate random numbers for each plane to simulate "hits" in the planes. Test of the track would have been "good". Do this N times and use the number of "good" tracks to compute the "track efficiency".
# * **road_efficiency_study(N)**: Test every roadID for every intensity setting N times and spit out the track efficiency for each (roadID, intensity) setting.


def get_mean_wires_from_hodo(stationID, detectorHalf, elementID):
    # Bottom-half only corresponds to D3 minus
    # Top-half only corresponds to D3 plus
    det_dict = {'B' : ("D3pU", "D3pUp", "D3pV", "D3pVp", "D3pX", "D3pXp"),
                'T' : ("D3mU", "D3mUp", "D3mV", "D3mVp", "D3mX", "D3mXp")}
    query = ("""hodoDetectorName == \"H%d%s\" and \
                hodoElementID==%d and \
                wireDetectorName not in @det_dict[@detectorHalf]""" %
             (stationID, detectorHalf, elementID))
    df = (hodo_mask
             .query(query)
             [['wireDetectorName', 'wireElementID']]
             .groupby('wireDetectorName')
             .mean()
             .round()
             .reset_index())
    return df


def get_mean_wires_from_road(roadID):
    hodos = (trigger_roads.query("roadID == @roadID")
                 [['detectorHalf', 'H1', 'H2', 'H3']]
                 .values[0])
    df = pd.DataFrame()
    for i in range(1,4):
        temp_df = get_mean_wires_from_hodo(i, hodos[0], hodos[i])
        df = pd.concat([df, temp_df])
    df = df.reset_index()
    df.drop('index', inplace=True, axis=1)
    return df


def get_efficiency(plane, element, intensity, target):
    query = """detectorName == @plane and \
               @element >= minElementID and \
               @element <= maxElementID and \
               intensity == @intensity and \
               target == @target"""
    res = chamber_eff.query(query)[['efficiency', 'uncertainty']].values[0]
    return (res[0], res[1])


def get_section(plane, element, intensity, target):
    query = """detectorName == @plane and \
               @element >= minElementID and \
               @element <= maxElementID and \
               intensity == @intensity and \
               target == @target"""
    res = chamber_eff.query(query)['section'].values[0]
    return res


def test_track(hits):
    
    c1 = ('D1U','D1V','D1X')
    c2 = ('D2U','D2V','D2X')
    c3 = ('D3U','D3V','D3X')
    c_all = c1 + c2 + c3
    
    # Require at least 15 hits
    if hits['hit'].sum() < 15:
        return False
    
    # Remove the m's and p's from the detectorNames
    hits['detectorName'] = (hits['detectorName']
                                .apply(lambda x: x.replace('m','').replace('p','')))
    
    # Require at least 4 hits in each station
    if hits[hits.detectorName.isin(c1)].sum().values[1] < 4.0:
        return False
    if hits[hits.detectorName.isin(c2)].sum().values[1] < 4.0:
        return False
    if hits[hits.detectorName.isin(c3)].sum().values[1] < 4.0:
        return False
    
    # Require at least one hit in each 'view' (prime-unprime pair)
    for c in c_all:
        if hits[hits.detectorName==c].sum().values[1] < 1.0:
            return False
        
    return True


def road_test(roadID, intensity, N, hit_dict, target):
    # Get the mean wireElementID covered by each paddle of the road
    #   for each of the 18 planes
    df = get_mean_wires_from_road(roadID)
    
    # Determine the 'section' of the chambers
    df['section'] =  \
       df.apply(lambda x: get_section(x.wireDetectorName, x.wireElementID, intensity, target),
                axis=1)
    
    good_tracks = 0.0
    hits = pd.DataFrame(columns=['detectorName', 'hit'])
    for i in range(0,N):
        hits.drop(hits.index, inplace=True)
        for det, section in zip(df.wireDetectorName, df.section):
            hits = hits.append({"detectorName": det,
                                "hit": hit_dict[(det, section, intensity)][i]},
                               ignore_index=True)
        
        if test_track(hits):
            good_tracks += 1.0
        
    track_eff = float(good_tracks) / float(N)
    track_eff_unc = np.sqrt(float(good_tracks)) / float(N)
    
    return track_eff, track_eff_unc


def simulate_hits(N):
    hit_dict = {}
    for row in chamber_eff.iterrows():
        hit_list = []
        hit = 0
        for i in range(0,N):
            my_eff = row[1].efficiency + np.random.normal(loc=0.0,
                                                          scale=row[1].uncertainty)
            if np.random.rand() < my_eff:
                hit = 1
            else:
                hit = 0
            hit_list.append(hit)
        hit_dict[(row[1].detectorName, row[1].section, row[1].intensity, row[1].target)] = hit_list
        
    return hit_dict


#%%time
#road_test(-77294, 15000, 300, hit_dict)
# 0.98 +/- 0.0990 # with 100 hits (9.37s)
# 0.97 +/- 0.0696  # with 200 hits (18.7s)
# 0.97 +/- 0.0566  # with 300 hits (27s)
# 0.975 +/- 0.0494 # with 400 hits (36.4s)
# 0.971 +/- 0.0312 # with 1000 hits (90s)


def road_efficiency_study(N, roads, target, filename):
    # Get list of intensities to use
    intensities = [5000+i*10000 for i in range(0,10)]
    
    # Generate the pile of simulated hits to use
    hit_dict = simulate_hits(N)
    
    # Set up data frame to hold results
    my_df = pd.DataFrame(columns=['roadID', 'intensity', 'efficiency', 'uncertainty'])
    
    # Test each roadID
    for road in roads:
        # ...for each intensity
        for intensity in intensities:
            eff, unc = road_test(road, intensity, N, hit_dict)
            with open(filename, 'a') as f:
                f.write("%i\t%d\t%f\t%f\n" % (road, intensity, eff, unc))
            my_df = my_df.append({'roadID': road,
                                  'intensity': intensity,
                                  'efficiency': eff,
                                  'uncertainty': unc}, ignore_index=True)
    return my_df


def main():
    opts = docopt.docopt(__doc__, version='Track Efficiency v0.1')
    print opts
    N = int(opts['-N'])
    if opts['--nsets'] and opts['--set']:
        nsets = int(opts['--nsets'])
        nset = int(opts['--set'])
    else:
        nsets = None
        nset = None
    if opts['--target']:
        target = int(opts['--target'])
    else:
        target = None
    
    chunk = len(trigger_roads.roadID) / nsets
    roads = set()
    
    if nset <= 0 or nsets <= 0 or nset > nsets:
	print "Error. Make sure that <set> and <nsets> are > 0 and that <set> <= <nsets>."
	return -1
    if nset < nsets:
	roads.update(set(trigger_roads
		             .roadID
                             .values
                             [(nset-1)*chunk:nset*chunk]
                             .tolist()))
    elif nset == nsets:
	roads.update(set(trigger_roads.roadID.values[(nset-1)*chunk:].tolist()))

    filename = ("data/road_eff_%d_of_%d_n_%d.tsv" % (nset, nsets, N))
    road_eff_df = road_efficiency_study(N, roads, filename)

    return 0
    


if __name__=='__main__':
    main()
#road_eff_df = road_efficiency_study(3, "road_eff_n_3.tsv")

