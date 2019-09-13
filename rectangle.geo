delta = 0.2;

//OMEGA RECTANGLE --------------------------------
Point(1) = {0, 0, 0, 1};
Point(2) = {1, 0, 0, 1};
Point(3) = {1, 1, 0, 1};
Point(4) = {0, 1, 0, 1};

Line(11) = {1,2};
Line(12) = {2,3};
Line(13) = {03,4};
Line(14) = {4,1};

Line Loop(111) = {11,12,13,14};
Physical Line("Bound_Omega", 9) = {11,12,13,14};

//OMEGA_I RECTANGLE --------------------------------
Point(5) = {-delta, -delta, 0, 1};
Point(6) = {1+delta, -delta, 0, 1};
Point(7) = {1+delta, 1+delta, 0, 1};
Point(8) = {-delta, 1+delta, 0, 1};

Line(15) = {5,6};
Line(16) = {6,7};
Line(17) = {7,8};
Line(18) = {8,5};

Line Loop(112) = {15,16,17,18};
Physical Line("Bound_OmegaI", 99) = {15,16,17,18};

//OMEGA (RECTANGLE SURFACE)-------------------------
Plane Surface(21) = {111};
Transfinite Surface {21};
Physical Surface("Omega", 1) = {21};

//OMGEA_I (RECTANGLE SURFACE) ----------------------
Plane Surface(22) = {112, 111};
Transfinite Surface {22};
Physical Surface("Omega_I", 2) = {22};


