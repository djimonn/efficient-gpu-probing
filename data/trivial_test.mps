* Simple probing test instance
* x0 + x1 <= 1
* both variables binary

NAME          PROBE1
ROWS
 N  OBJ
 L  C1
COLUMNS
    MARK0000  'MARKER'                 'INTORG'
    X0        OBJ                  0   C1                   1
    X1        OBJ                  0   C1                   1
    MARK0001  'MARKER'                 'INTEND'
RHS
    RHS1      C1                   1
BOUNDS
 BV BND1      X0
 BV BND1      X1
ENDATA