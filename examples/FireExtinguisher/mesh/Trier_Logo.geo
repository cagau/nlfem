lc = 1;

//------------------------------------------------------------------------------
// OMEGA
Point(1) = {7/10, 0, 0,  lc};
Point(2) = {7/10, 11/10, 0,  lc};
Point(3) = {0, 11/10, 0,  lc};
Point(4) = {0, 8.5/10, 0,  lc};
Point(5) = {4.2/10, 8.5/10, 0,  lc};
Point(6) = {4.2/10, 0.8/10, 0,  lc};
Point(11) = {5.2/10, 0.3/10,0,  lc};

// Boundary of Omega
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,5};
Line(5) = {5,6};
Spline(6) = {6,11,1};
Line Loop(8) = {1,2,3,4,5,6};
Physical Line(88) = {8};

// Interior of Omega
Plane Surface(11) = {8};
Physical Surface(99) = {11};



// Omega_I
Point(7) = {3.3/10, 1.3/10,0, lc};
Point(8) = {3.3/10, 7.5/10, 0, lc};
Point(9) = {0, 7.5/10, 0, lc};
Point(10) = {1/10, 4/10, 0, lc};

// Boundary of Omega_I
Line(9) = {7, 8};
Line(10) = {8, 9};
Spline(11) = {9, 10,7};
Line Loop(12) = {9, 10, 11};
Physical Line(9) = {1, 2, 3, 4};

// Interior of Omega_I
Plane Surface(22) = {12};
Physical Surface(77) = {22};

Mesh.MshFileVersion = 2.2;


