from enum import IntEnum


class TotalSpineSegLabels(IntEnum):
    """
    Segmentation labels according to the labeling used by TotalSpineSeg.
    """

    SPINAL_CORD = 1
    SPINAL_CANAL = 2

    VERTEBRAE_C1 = 11
    VERTEBRAE_C2 = 12
    VERTEBRAE_C3 = 13
    VERTEBRAE_C4 = 14
    VERTEBRAE_C5 = 15
    VERTEBRAE_C6 = 16
    VERTEBRAE_C7 = 17

    VERTEBRAE_T1 = 21
    VERTEBRAE_T2 = 22
    VERTEBRAE_T3 = 23
    VERTEBRAE_T4 = 24
    VERTEBRAE_T5 = 25
    VERTEBRAE_T6 = 26
    VERTEBRAE_T7 = 27
    VERTEBRAE_T8 = 28
    VERTEBRAE_T9 = 29
    VERTEBRAE_T10 = 30
    VERTEBRAE_T11 = 31
    VERTEBRAE_T12 = 32

    VERTEBRAE_L1 = 41
    VERTEBRAE_L2 = 42
    VERTEBRAE_L3 = 43
    VERTEBRAE_L4 = 44
    VERTEBRAE_L5 = 45

    SACRUM = 50

    DISC_C2_C3 = 63
    DISC_C3_C4 = 64
    DISC_C4_C5 = 65
    DISC_C5_C6 = 66
    DISC_C6_C7 = 67

    DISC_C7_T1 = 71
    DISC_T1_T2 = 72
    DISC_T2_T3 = 73
    DISC_T3_T4 = 74
    DISC_T4_T5 = 75
    DISC_T5_T6 = 76
    DISC_T6_T7 = 77
    DISC_T7_T8 = 78
    DISC_T8_T9 = 79
    DISC_T9_T10 = 80
    DISC_T10_T11 = 81
    DISC_T11_T12 = 82

    DISC_T12_L1 = 91
    DISC_L1_L2 = 92
    DISC_L2_L3 = 93
    DISC_L3_L4 = 94
    DISC_L4_L5 = 95
    DISC_L5_S = 100
