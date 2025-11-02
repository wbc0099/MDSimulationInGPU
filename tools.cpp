#include "define.h"
#include "math.h"

__host__ __device__ int sign(real x){
    return -(x<0.f)+(x>0.f);
}

__host__ __device__ int sign01(real x){
    return (sign(x)+1)/2;
}

real distancePBC(real x0, real y0, real x1, real y1, real lx, real ly){
    real dx=sign(x1-x0)*(x1-x0);
    real dy=sign(y1-y0)*(y1-y0);
    dx = sign01(0.5 * lx - dx) * dx + sign01(dx - 0.5 * lx) * (lx - dx);
    dy = sign01(0.5 * ly - dy) * dy + sign01(dy - 0.5 * ly) * (ly - dy);
    real dr2=dx*dx+dy*dy;
    return sqrt(dr2);
}

void movePositionPBC(real x, real y, real dx, real dy, real lx, real ly, real& xNew, real& yNew){
    xNew=fmod(x+dx+lx,lx);
    yNew=fmod(y+dy+ly,ly);
}

