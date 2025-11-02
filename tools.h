void movePositionPBC(real x, real y, real dx, real dy, real lx, real ly, real& xNew, real& yNew);
real distancePBC(real x0, real y0, real x1, real y1, real lx, real ly);
__host__ __device__ int sign01(real x);
__host__ __device__ int sign(real x);