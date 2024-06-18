#include <iostream>
#include <vector>
#include <array>
#include <easyx.h>
#include <math.h>
#include <time.h>

using namespace std;
struct connection{};
struct spring:public connection
{
    double k;
    double L;
    double dL;
    vector<double> F(){
        return {dL*k,0};
    }
};

struct damper:public connection
{
    double f;
    double L;
    vector<double> F(double v){
        return {f*v,0};
    };
};

class matter{
    double m;
    POINT pos;
    POINT v; 
};

double length(POINT a,POINT b)
{
    return sqrt(pow(a.x-b.x,2)+pow(a.y-b.y,2));
}

int main()
{
    double m1=1,m2=20;
    POINT m1p={300,300};
    POINT m2p={300,400};
    POINT m1v={0,0};
    POINT m2v={0,0};
    POINT F1,F2,a1,a2;
    double f=1,k=1;
    double L=100;
    double dL=0;
    timeb*t,*t1;
    ftime(t);
    while (1)
    {
        dL=length(m1p,m2p)-L;
    


        ftime(t1);
        double dt=(t1->millitm-t->millitm)/1000.+(t1->time-t->time);
        t=t1;
    }
}