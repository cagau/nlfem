delta = 0.1;
cl__1 = 1;
lc = 1;
lc2 = 1;
lc3 = 1;
lc_fine = lc/3;
lc_point= lc/(9);
//------------------------------------------------------------------------------
// Interface
Point(12) = {0.5, 1.0, 0, lc};
Point(13) = {0.45, 0.75, 0, lc};
Point(15) = {0.55, 0.25, 0, lc};
Point(16) = {0.5, 0.0, 0, lc};

// Interface
Point(17) = {0.0, 0.5, 0, lc};
Point(18) = {0.25, 0.45, 0, lc};
Point(19) = {0.75, 0.55, 0, lc};
Point(20) = {1.0, 0.5, 0, lc};

Point(99) = {0.5, 0.5, 0, lc};

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
Line(1) = {2,17};
Line(19) = {17,1};
Line(2) = {2,12};
Line(3) = {12,3};
Line(4) = {3, 20};
Line(49) = {20, 4};
Line(5) = {4,16};
Line(6) = {16, 1};
Line(7) = {11, 9};
Line(8) = {9, 8};
Line(9) = {8, 10};
Line(10) = {10, 11};

//================ Interfaces ================//
Spline(1122) = {17, 18, 99};
Spline(2233) = {16, 15, 99};
Spline(3344) = {20,19,99};
Spline(4411) = {12, 13,99};

//================ Add Surfaces ================//
// Omega_1
Line Loop(12) = {1, 1122, -4411, -2};
Plane Surface(13) = {12};

// Omega_2
Line Loop(18) = {1122, -2233, 6, -19};
Plane Surface(19) = {18};

// Omega_3
Line Loop(19) = {2233, -3344, 49, 5};
Plane Surface(20) = {19};

// Omega_4
Line Loop(20) = {3344, -4411, 3, 4};
Plane Surface(21) = {20};


// Omega_I
Line Loop(17) = {10, 7, 8, 9};
Line Loop(333) = {19, -6, -5, -49, -4, -3, -2, 1};

Plane Surface(18) = {17, 333};



//=============== LABELING ===============//
// boundary of Omega
Physical Line(99) = {1, 6, 5, 4, 3, 2};

// Interface
Physical Line(12) = {1122};

// boundary of Omega_1
Physical Line(11) = {1, 1122, 4411, 2};

// boundary of Omega_2
Physical Line(22)= {19, 1122, 2233, 6};

// boundary of Omega_3
Physical Line(33) = {2233, 3344, 49, 5};

// boundary of Omega_4
Physical Line(44) = {4411, 3344, 4, 3};

// Omega_1
Physical Surface(1) = {13};

// Omega_2
Physical Surface(2) = {19};

// Omega_3
Physical Surface(3) = {20};

// Omega_4
Physical Surface(4) = {21};

// Omega_I
Physical Surface(5) = {18};


