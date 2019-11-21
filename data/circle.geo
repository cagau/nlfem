delta = 0.3;
cl__1 = 1;

Point(1) = {0, 0, 0, 1};

// OMEGA CIRCLE --------------------------------
Point(2) = {-0, 1, 0, 1};
Point(3) = {0, -1, 0, 1};
Circle(1) = {2, 1, 3};
Circle(2) = {3, 1, 2};


// OMEGA_I CIRCLE ------------------------------
Point(4) = {-1.0 - delta, 0, 0, 1};
Point(5) = {1.0 + delta, 0, 0, 1};
Circle(5) = {4, 1, 5};
Circle(6) = {5, 1, 4};


// OMEGA ---------------------------------------
Line Loop(4) = {1, 2};
Plane Surface(4) = {4};
Physical Surface(1) = {4};

//boundary of omega
Physical Line(9) = {1, 2};

// OMEGA_I -------------------------------------
Line Loop(13) = {1, 2, -5, -6};
Plane Surface(13) = {13};
Physical Surface(2) = {13};

// boundary of omega_i
Physical Line(99) = {5, 6};


