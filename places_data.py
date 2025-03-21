from dataclasses import dataclass

@dataclass
class Place():
    mean_bad_scores: float
    std_dev_bad_scores: float
    filter_n: float

# Place-level averaged values (from place_averages_oldDB.csv)
places = [
    Place(mean_bad_scores=0.02939746, std_dev_bad_scores=0.021945703, filter_n=12.31531987),  # p0
    Place(mean_bad_scores=0.033859191, std_dev_bad_scores=0.024805636, filter_n=8.706349206),  # p1
    Place(mean_bad_scores=0.051930638, std_dev_bad_scores=0.028516954, filter_n=2.809983292),   # p2
    Place(mean_bad_scores=0.038821565, std_dev_bad_scores=0.035071892, filter_n=4.282407407),   # p3
    Place(mean_bad_scores=0.060289883, std_dev_bad_scores=0.058448452, filter_n=4.383333333),   # p4
    Place(mean_bad_scores=0.050924457, std_dev_bad_scores=0.047838804, filter_n=3.848684211),   # p5
    Place(mean_bad_scores=0.06233081, std_dev_bad_scores=0.05963089, filter_n=4.548393932),   # p6
    Place(mean_bad_scores=0.073274606, std_dev_bad_scores=0.065840657, filter_n=3.342815372),   # p7
    Place(mean_bad_scores=0.046530576, std_dev_bad_scores=0.034476063, filter_n=4.398991228),   # p8
    Place(mean_bad_scores=0.029912096, std_dev_bad_scores=0.023423167, filter_n=6.77830941)    # p9
]

# Individual image-level values for each place (from image_averages_oldDB.csv)
places_matrix = [
    [
        Place(mean_bad_scores=0.023182337, std_dev_bad_scores=0.015629124, filter_n=17.27777778),  # p0/i0
        Place(mean_bad_scores=0.022056218, std_dev_bad_scores=0.015475444, filter_n=17.35),  # p0/i1
        Place(mean_bad_scores=0.042953826, std_dev_bad_scores=0.034732541, filter_n=2.318181818)    # p0/i2
    ],
    [
        Place(mean_bad_scores=0.031925956, std_dev_bad_scores=0.021655685, filter_n=11.19047619),  # p1/i0
        Place(mean_bad_scores=0.029995171, std_dev_bad_scores=0.02106008, filter_n=12.16666667),  # p1/i1
        Place(mean_bad_scores=0.039656447, std_dev_bad_scores=0.031701143, filter_n=2.761904762)    # p1/i2
    ],
    [
        Place(mean_bad_scores=0.059134482, std_dev_bad_scores=0.028666724, filter_n=2.45),   # p2/i0
        Place(mean_bad_scores=0.048412272, std_dev_bad_scores=0.027839112, filter_n=3.19047619),   # p2/i1
        Place(mean_bad_scores=0.04824516, std_dev_bad_scores=0.029045025, filter_n=2.789473684)   # p2/i2
    ],
    [
        Place(mean_bad_scores=0.047722718, std_dev_bad_scores=0.04581057, filter_n=2.055555556),   # p3/i0
        Place(mean_bad_scores=0.029263525, std_dev_bad_scores=0.025968407, filter_n=6.833333333),   # p3/i1
        Place(mean_bad_scores=0.039478453, std_dev_bad_scores=0.0334367, filter_n=3.958333333)    # p3/i2
    ],
    [
        Place(mean_bad_scores=0.067845809, std_dev_bad_scores=0.059069832, filter_n=5.0),   # p4/i0
        Place(mean_bad_scores=0.052025856, std_dev_bad_scores=0.054761489, filter_n=4.95),   # p4/i1
        Place(mean_bad_scores=0.060997983, std_dev_bad_scores=0.061514036, filter_n=3.2)   # p4/i2
    ],
    [
        Place(mean_bad_scores=0.066937238, std_dev_bad_scores=0.04854081, filter_n=2.0),   # p5/i0
        Place(mean_bad_scores=0.042206538, std_dev_bad_scores=0.042758301, filter_n=5.125),   # p5/i1
        Place(mean_bad_scores=0.043629593, std_dev_bad_scores=0.052217299, filter_n=4.421052632)    # p5/i2
    ],
    [
        Place(mean_bad_scores=0.064813917, std_dev_bad_scores=0.05896997, filter_n=4.473684211),   # p6/i0
        Place(mean_bad_scores=0.049141572, std_dev_bad_scores=0.055136599, filter_n=4.388888889),  # p6/i1
        Place(mean_bad_scores=0.073036941, std_dev_bad_scores=0.064786102, filter_n=4.782608696)   # p6/i2
    ],
    [
        Place(mean_bad_scores=0.070509899, std_dev_bad_scores=0.067458392, filter_n=3.65),   # p7/i0
        Place(mean_bad_scores=0.071921157, std_dev_bad_scores=0.063376914, filter_n=3.904761905),  # p7/i1
        Place(mean_bad_scores=0.077392762, std_dev_bad_scores=0.066686665, filter_n=2.473684211)    # p7/i2
    ],
    [
        Place(mean_bad_scores=0.04432663, std_dev_bad_scores=0.028928927, filter_n=4.6875),   # p8/i0
        Place(mean_bad_scores=0.044364564, std_dev_bad_scores=0.027535195, filter_n=5.789473684),   # p8/i1
        Place(mean_bad_scores=0.050900534, std_dev_bad_scores=0.046964068, filter_n=2.72)   # p8/i2
    ],
    [
        Place(mean_bad_scores=0.037482211, std_dev_bad_scores=0.023687797, filter_n=3.545454545),   # p9/i0
        Place(mean_bad_scores=0.028130576, std_dev_bad_scores=0.026027564, filter_n=7.631578947),  # p9/i1
        Place(mean_bad_scores=0.024123502, std_dev_bad_scores=0.020554141, filter_n=9.157894737)   # p9/i2
    ]
] 