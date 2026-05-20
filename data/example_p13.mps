* Example for iterative probing
* See "Preprocessing and Probing Techniques for Mixed Integer Programming Problems" by M.W.P. Savelsbergh, p.13
NAME          EXAMPLE
ROWS
 N  OBJ
 G  DEM1
 G  DEM2
 G  DEM3
 L  CAP1
 L  CAP2
 L  CAP3
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    X1        OBJ                 24   CAP1               -15
    X2        OBJ                 12   CAP2               -20
    X3        OBJ                 16   CAP3                -5
    MARK0001  'MARKER'                 'INTEND'
    Y1        OBJ                  4   DEM1                 1
    Y1        DEM2                 1   DEM3                 2
    Y1        CAP1                 1
    Y2        OBJ                  2   DEM1                 3
    Y2        DEM3                 1   CAP2                 1
    Y3        OBJ                  3   DEM2                 2
    Y3        CAP3                 1
RHS
    RHS1      DEM1                15   DEM2                10
    RHS1      DEM3                20   CAP1                 0
    RHS1      CAP2                 0   CAP3                 0
BOUNDS
 BV BND1      X1
 BV BND1      X2
 BV BND1      X3
 LO BND1      Y1                  0
 LO BND1      Y2                  0
 LO BND1      Y3                  0
ENDATA