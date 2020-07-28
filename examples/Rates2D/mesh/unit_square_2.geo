delta = 0.1;
cl__1 = 1;
lc = 1;
lc_fine = lc/3;
lc_coarse = lc;


//Point for coarsening
Point(555) = {0.5, 0.55, 0.0, lc_coarse};
Point(5552) = {0.2, 0.85, 0.0, lc_coarse};
Point(5553) = {0.8, 0.25, 0.0, lc_coarse};
Point(5554) = {0.2, 0.55, 0.0, lc_coarse};
Point(5555) = {0.4, 0.8, 0.0, lc_coarse};
Point(5556) = {0.15, 0.2, 0.0, lc_coarse};

// Point for refinement (in the center of the shape)
Point(99991) = {0.75, 0.76, 0.0, lc};
Point(99992) = {0.25, 0.25, 0.0, lc};
Point(99993) = {0.8, 0.5, 0.0, lc};
Point(99994) = {0.8, 0.1, 0.0, lc};

// Ellipse
SetFactory("OpenCASCADE");
Ellipse(8889) = {0.3, 0.7, 0, 0.2, 0.1 , 0, 2*Pi};
Point(777) = {0.25, 0.7, 0, lc_coarse};
Point(7772) = {0.35, 0.7, 0, lc_coarse};
Point(7773) = {0.4, 0.72, 0, lc_coarse};

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

// Ellipse
Line Loop(88882) = {8889};
Plane Surface(20) = {88882};

Point{777} In Surface {20};
Point{7772} In Surface {20};
Point{7773} In Surface {20};

// Omega
Line Loop(16) = {4, 1, 2, 3};
Plane Surface(17) = {16, 88882}; 

Point{555} In Surface {17};
Point{5552} In Surface {17};
Point{5553} In Surface {17};
Point{5554} In Surface {17};
Point{5555} In Surface {17};
Point{5556} In Surface {17};

// Omega_I
Line Loop(18) = {9, 6, 7, 8};
Plane Surface(19) = {16, 18};

//=============== LABELING ===============//
// boundary of Omega = label 9
Physical Line(9) = {1, 2, 3, 4};
// boundary of Omega_I = label 13
Physical Line(13) = {6, 7, 8, 9};

//Ellipse
Physical Surface(1) = {20};

// Omega = label 1
Physical Surface(2) = {17};

// Omega_I = label 2
Physical Surface(3) = {19};
 







// POINT ATTRACTORS

// Refinement around point
Field[1] = Attractor;
Field[1].EdgesList = {8889};
Field[1].NNodesByEdge = 1000;
Field[1].NodesList = {99991,99992, 99993,99994};

Field[2] = Threshold;
Field[2].IField = 1;
Field[2].LcMin = lc_fine;// element size inside DistMin
Field[2].LcMax = lc;     // element size outside DistMax
Field[2].DistMin = 0.05;
Field[2].DistMax = 0.1;

// Define minimum of threshold and function field
Field[3] = Min;
Field[3].FieldsList = {2};


// Use the min as the background field
Background Field = 3;



