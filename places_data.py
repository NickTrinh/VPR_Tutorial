from dataclasses import dataclass

@dataclass
class Place():
    mean_bad_scores: float
    std_dev_bad_scores: float
    filter_n: float

# Place-level averaged values
places = [
    Place(mean_bad_scores=0.027348, std_dev_bad_scores=0.021431, filter_n=21.266667),  # p0
    Place(mean_bad_scores=0.035565, std_dev_bad_scores=0.026386, filter_n=10.633333),  # p1
    Place(mean_bad_scores=0.049672, std_dev_bad_scores=0.030017, filter_n=3.416667),   # p2
    Place(mean_bad_scores=0.041013, std_dev_bad_scores=0.036605, filter_n=5.166667),   # p3
    Place(mean_bad_scores=0.060161, std_dev_bad_scores=0.057331, filter_n=5.444444),   # p4
    Place(mean_bad_scores=0.052739, std_dev_bad_scores=0.048123, filter_n=4.500000),   # p5
    Place(mean_bad_scores=0.045892, std_dev_bad_scores=0.040217, filter_n=6.733333),   # p6
    Place(mean_bad_scores=0.048021, std_dev_bad_scores=0.042843, filter_n=4.600000),   # p7
    Place(mean_bad_scores=0.053657, std_dev_bad_scores=0.049021, filter_n=4.500000),   # p8
    Place(mean_bad_scores=0.056789, std_dev_bad_scores=0.050902, filter_n=5.200000)    # p9
]

# Individual image-level values for each place
places_matrix = [
    [
        Place(mean_bad_scores=0.020714, std_dev_bad_scores=0.015921, filter_n=35.0),  # p0/i0
        Place(mean_bad_scores=0.022136, std_dev_bad_scores=0.016253, filter_n=24.8),  # p0/i1
        Place(mean_bad_scores=0.039192, std_dev_bad_scores=0.032119, filter_n=4.0)    # p0/i2
    ],
    [
        Place(mean_bad_scores=0.034049, std_dev_bad_scores=0.020597, filter_n=17.4),  # p1/i0
        Place(mean_bad_scores=0.031653, std_dev_bad_scores=0.023510, filter_n=11.0),  # p1/i1
        Place(mean_bad_scores=0.040993, std_dev_bad_scores=0.035052, filter_n=3.5)    # p1/i2
    ],
    [
        Place(mean_bad_scores=0.057334, std_dev_bad_scores=0.029992, filter_n=2.5),   # p2/i0
        Place(mean_bad_scores=0.045119, std_dev_bad_scores=0.028257, filter_n=5.0),   # p2/i1
        Place(mean_bad_scores=0.046564, std_dev_bad_scores=0.031802, filter_n=2.75)   # p2/i2
    ],
    [
        Place(mean_bad_scores=0.048083, std_dev_bad_scores=0.049731, filter_n=2.0),   # p3/i0
        Place(mean_bad_scores=0.032191, std_dev_bad_scores=0.028965, filter_n=8.0),   # p3/i1
        Place(mean_bad_scores=0.042766, std_dev_bad_scores=0.031119, filter_n=5.5)    # p3/i2
    ],
    [
        Place(mean_bad_scores=0.068348, std_dev_bad_scores=0.061148, filter_n=5.0),   # p4/i0
        Place(mean_bad_scores=0.053499, std_dev_bad_scores=0.054164, filter_n=7.0),   # p4/i1
        Place(mean_bad_scores=0.058637, std_dev_bad_scores=0.056680, filter_n=4.33)   # p4/i2
    ],
    [
        Place(mean_bad_scores=0.067613, std_dev_bad_scores=0.048807, filter_n=2.5),   # p5/i0
        Place(mean_bad_scores=0.040378, std_dev_bad_scores=0.041240, filter_n=7.0),   # p5/i1
        Place(mean_bad_scores=0.041706, std_dev_bad_scores=0.047086, filter_n=4.0)    # p5/i2
    ],
    [
        Place(mean_bad_scores=0.064587, std_dev_bad_scores=0.059426, filter_n=7.5),   # p6/i0
        Place(mean_bad_scores=0.043784, std_dev_bad_scores=0.052050, filter_n=6.33),  # p6/i1
        Place(mean_bad_scores=0.078152, std_dev_bad_scores=0.068723, filter_n=6.67)   # p6/i2
    ],
    [
        Place(mean_bad_scores=0.068702, std_dev_bad_scores=0.067478, filter_n=5.0),   # p7/i0
        Place(mean_bad_scores=0.069520, std_dev_bad_scores=0.062152, filter_n=5.67),  # p7/i1
        Place(mean_bad_scores=0.076234, std_dev_bad_scores=0.065766, filter_n=3.0)    # p7/i2
    ],
    [
        Place(mean_bad_scores=0.041822, std_dev_bad_scores=0.030725, filter_n=7.0),   # p8/i0
        Place(mean_bad_scores=0.042892, std_dev_bad_scores=0.026950, filter_n=7.0),   # p8/i1
        Place(mean_bad_scores=0.048624, std_dev_bad_scores=0.046853, filter_n=2.67)   # p8/i2
    ],
    [
        Place(mean_bad_scores=0.039208, std_dev_bad_scores=0.023883, filter_n=5.0),   # p9/i0
        Place(mean_bad_scores=0.029648, std_dev_bad_scores=0.024540, filter_n=10.5),  # p9/i1
        Place(mean_bad_scores=0.024691, std_dev_bad_scores=0.020590, filter_n=12.5)   # p9/i2
    ]
] 