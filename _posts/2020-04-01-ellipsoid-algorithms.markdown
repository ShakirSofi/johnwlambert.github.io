




## Ellipsoids

Cutting plane algorithm
Academic version center of gravity, only in 2 dimensions

For tens of years simplex was sacred. Then Khachiyan replaced it with ellipsoid algorithm. Worst case of simplex is disaster.  Ellipsoid improved it by guaranteeing polynomial time. In 1984 there was Karmarkar, provably polynomial, competitive with Simplex in real-life.
Interior-point method. Ellipsoid algorithm is not competitive at all with simplex method. Simplex is exponential in the dimension. Move from vertex to neighboring vertex. simplex is Family of algorihtms, different wayss to move to neighboring vertex. Practical use of Simplex also depends on sparsity, as well as dimension. 10,000s of variables and constraints. Default version today is interior point.


## Affine image of unit ball u
Bu + c, c is center of ellipsoid


## Cut ellipsoid in half


function of single variable to cover it with another ellipsoid


1. Nemirovski transparentices ISyE 6663.

2.  Khachiyan, L. G. 1979. "A Polynomial Algorithm in Linear Programming". Doklady Akademii Nauk SSSR 244, 1093-1096 (translated in Soviet Mathematics Doklady 20, 191-194, 1979). [PDF (English)](https://www.sciencedirect.com/science/article/abs/pii/0041555380900610). [PDF (In Russian for those of us who speak it)](http://www.mathnet.ru/links/438aedbe7d2dbb2dbc5c176025f61cb6/dan42319.pdf).

2. Robert Bland, Donald Goldfarb, Michael Todd. The Ellipsoid Method: A Survey. [PDF](
http://www.math.uwaterloo.ca/~cswamy/courses/co759/approx-material/ellipsoid-survey.pdf).