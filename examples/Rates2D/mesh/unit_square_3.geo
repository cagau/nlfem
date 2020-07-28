delta = 0.1;
cl__1 = 1;
lc = 1;
lc_fine = lc/2.5;


// Point for refinement (in the center of the shape)
Point(99991) = {0.0, 0.0, 0.0, lc};
Point(99992) = {0.5, 0.5, 0.0, lc};
//Point(99993) = {0.4, 0.5, 0.0, lc};
//Point(99994) = {0.8, 0.1, 0.0, lc};

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
Line(7) = {9, 8};
Line(8) = {8, 10};
Line(9) = {10, 11};

// Omega
Line Loop(16) = {4, 1, 2, 3};
Plane Surface(17) = {16}; 
// Omega_I
Line Loop(18) = {9, 6, 7, 8};
Plane Surface(19) = {16, 18};

//=============== LABELING ===============//
// boundary of Omega = label 9
Physical Line(9) = {1, 2, 3, 4};
// boundary of Omega_I = label 13
Physical Line(13) = {6, 7, 8, 9};
// Omega = label 1
Physical Surface(1) = {17};
// Omega_I = label 2
Physical Surface(2) = {19};
 


// POINT ATTRACTORS

// Refinement around point
Field[1] = Attractor;
//Field[1].EdgesList = {10};

Field[1].NodesList = {99991,99992, 99993,99994};

Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = lc_fine;// element size inside DistMin
Field[2].LcMax = lc;     // element size outside DistMax
Field[2].DistMin = 0.05;
Field[2].DistMax = 0.2;

// Define minimum of threshold and function field
Field[3] = Min;
Field[3].FieldsList = {2};


// Use the min as the background field
Background Field = 3;





