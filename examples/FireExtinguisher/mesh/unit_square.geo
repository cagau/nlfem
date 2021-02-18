delta = 0.1;
cl__1 = 1;
lc = 1;

//------------------------------------------------------------------------------
// OMEGA
Point(1) = {0, 0, 0,  lc};
Point(2) = {0, 1, 0,  lc};
Point(3) = {1, 1, 0,  lc};
Point(4) = {1, 0, 0,  lc};

// Omega_I
Point(8) = {1. + delta, -delta, -0, lc};
Point(9) = {-delta, -delta, -0, lc};
Point(10) = {1 + delta, 1 + delta, -0, lc};
Point(11) = {-delta, 1 + delta, -0, lc};

// boundary of Omega
Line(1) = {2, 1};
Line(2) = {1, 4};
Line(3) = {4, 3};
Line(4) = {3, 2};

// boundary of Omega_I
Line(6) = {11, 9};
//Transfinite Curve{6} = 200;
Line(7) = {9, 8};
//Transfinite Curve{7} = 200;
Line(8) = {8, 10};
//Transfinite Curve{7} = 200;
Line(9) = {10, 11};
//Transfinite Curve{9} = 200;

// Omega
Line Loop(16) = {4, 1, 2, 3};
Plane Surface(17) = {16}; 
Transfinite Surface(17) = {1, 2, 3, 4};

// for quadrilaterals use:
//Recombine Surface(17);

// Omega_I
Line Loop(18) = {9, 6, 7, 8};
Plane Surface(19) = {16, 18};
//Transfinite Surface(19) = {8,9,10,11};




//=============== LABELING ===============//
// boundary of Omega = label 9
Physical Line(9) = {1, 2, 3, 4};

// boundary of Omega_I = label 13
Physical Line(13) = {6, 7, 8, 9};

// Omega = label 1
Physical Surface(1) = {17};

// Omega_I = label 2
Physical Surface(92) = {19};


Mesh 2;
Save "unit_square.msh";
