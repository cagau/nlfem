delta = 0.4;
cl__1 = 1;
lc = 1;
lc2 = 1;
lc3 = 1;
lc_fine = lc/3;
lc_point= lc/(9);
//------------------------------------------------------------------------------
// Interface
Point(12) = {0.5, 1.0, 0, lc};
Point(13) = {0.35, 0.75, 0, lc};
Point(15) = {0.65, 0.25, 0, lc};
Point(16) = {0.5, 0.0, 0, lc};

//------------------------------------------------------------------------------
// OMEGA
Point(1) = {0, 0, 0,  lc};
Point(2) = {0, 1, 0,  lc};
Point(3) = {1, 1, 0,  lc};
Point(4) = {1, 0, 0,  lc};

// OMEGA_I
Point(8) = {1 + delta, -delta, -0, lc3};
Point(9) = {-delta, -delta, -0, lc3};
Point(10) = {1 + delta, 1 + delta, -0, lc3};
Point(11) = {-delta, 1 + delta, -0, lc3};

//------------------------------------------------------------------------------
Line(1) = {1,2};
Line(2) = {2,12};
Line(3) = {12,3};
Line(4) = {3, 4};
Line(5) = {4,16};
Line(6) = {16, 1};
Line(7) = {11, 9};
Line(8) = {9, 8};
Line(9) = {8, 10};
Line(10) = {10, 11};

//================ Interface ================//
Spline(11) = {12, 13, 15, 16};

//================ Add Surfaces ================//
// Omega_1
Line Loop(12) = {1, 2, 11, 6};
Plane Surface(13) = {12};
// Omega_2
Line Loop(14) = {11, -5, -4, -3};
Plane Surface(15) = {14};

// Omega_I
Line Loop(16) = {2, 3, 4, 5, 6, 1};
Line Loop(17) = {10, 7, 8, 9};
Plane Surface(18) = {16, 17};

//=============== LABELING ===============//
// boundary of Omega
Physical Line(1122) = {1, 6, 5, 4, 3, 2};

// Interface
Physical Line(12) = {11};

// boundary of Omega_1
Physical Line(11) = {1, 2,6};

// boundary of Omega_2
Physical Line(22) = {5, 3, 4};

// Omega_1
Physical Surface(1) = {13};

// Omega_2
Physical Surface(2) = {15};

// Omega_I
Physical Surface(3) = {18};


