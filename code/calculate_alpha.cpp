
/* calculate alpha from t=2 to T, in forward_bacward_hidden_unit_optimized.
 * syntax:       [last_exp_coef, last_exp_coef1, last_exp_coef2] = calculate_alpha(alpha, alpha_exp_coef, psi, last_exp_coef1, last_exp_coef2, T, H)
 * since we do not use const pointer, hence the value of alpha and alpha_exp_coef is updated automated. 
 *
 */




#include <math.h>
#include "mex.h"
#include "matrix.h"

//extern void _main();

double get_max_psi_at_t(double* psi, double* alpha, int t, int index1, int index2, int T, int H)
{
    double rtn = psi[t+T*( 0+ H*(index1 + 2*index2) )] + alpha[index1+2*(0+H*(t-1))];
    
    for(int i=1; i<H; i++)
    {
        if(rtn < psi[t+T*( i+ H*(index1 + 2*index2) )] + alpha[index1+2*(i+H*(t-1))])
            rtn = psi[t+T*( i+ H*(index1 + 2*index2) )] + alpha[index1+2*(i+H*(t-1))];
    }
    
    return rtn;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
    /* Check inputs */
    if(nrhs != 7)
        mexErrMsgTxt("Function should have seven inputs.");
    
    /* get inputs */
    double* alpha_initial = mxGetPr(prhs[0]);
    double* alpha_exp_coef_initial = mxGetPr(prhs[1]);
    double* psi = mxGetPr(prhs[2]);
    double last_exp_coef1 = mxGetScalar(prhs[3]);
    double last_exp_coef2 = mxGetScalar(prhs[4]);
    int T = (int)mxGetScalar(prhs[5]);
    int H = (int)mxGetScalar(prhs[6]);
    
//     mexPrintf("T = %d, H = %d, lastcoef1 = %f \n", T, H, last_exp_coef1);
    
    int N[3];
    N[0] = 2;
    N[1] = H;
    N[2] = T;
    plhs[3] = mxCreateNumericArray(3, N, mxDOUBLE_CLASS, mxREAL);
    plhs[3] = mxDuplicateArray(prhs[0]);
    double* alpha = mxGetPr(plhs[3]);
    plhs[4] = mxCreateNumericArray(3, N, mxDOUBLE_CLASS, mxREAL);
    plhs[4] = mxDuplicateArray(prhs[1]);
    double* alpha_exp_coef = mxGetPr(plhs[4]);
    
    
    /* calculate alpha by performing forward pass */
    double max_last_exp_coef = (last_exp_coef1 > last_exp_coef2) ? last_exp_coef1 : last_exp_coef2;
    double pa1, pa2, max_pa;

    for(int t=1; t<T; t++)
    {
        double exp_coef1 = max_last_exp_coef;
        double max_temp1 = get_max_psi_at_t(psi, alpha, t, 0, 1, T, H);
        double max_temp2 = get_max_psi_at_t(psi, alpha, t, 1, 1, T, H);
        double exp_coef2 = (max_temp1 + last_exp_coef1 > max_temp2 + last_exp_coef2) ? (max_temp1 + last_exp_coef1) : (max_temp2 + last_exp_coef2);
        for(int h=0; h<H; h++)
        {
            pa1 = alpha[2*(h+H*(t-1))] + last_exp_coef1 - exp_coef1;
            pa2 = alpha[1+2*(h+H*(t-1))] + last_exp_coef2 - exp_coef1;
//             alpha[2*(h+H*t)] = log(exp(pa1) + exp(pa2));      
            max_pa = (pa1 > pa2) ? pa1 : pa2;
            alpha[2*(h+H*t)] = max_pa + log(exp(pa1-max_pa) + exp(pa2-max_pa));      
            pa1 = psi[t+T*(h+H*(2*1))] - exp_coef2 + alpha[2*(h+H*(t-1))] + last_exp_coef1;
            pa2 = psi[t+T*(h+H*(1+2*1))] - exp_coef2 + alpha[1+2*(h+H*(t-1))] + last_exp_coef2;
//             alpha[1+2*(h+H*t)] = log(exp(pa1) + exp(pa2));           
            max_pa = (pa1 > pa2) ? pa1 : pa2;
            alpha[1+2*(h+H*t)] = max_pa + log(exp(pa1-max_pa) + exp(pa2-max_pa));      
        }
        last_exp_coef1 = exp_coef1;
        last_exp_coef2 = exp_coef2;
        max_last_exp_coef = (last_exp_coef1 > last_exp_coef2) ? last_exp_coef1 : last_exp_coef2;
        for(int h=0; h<H; h++)
        {
            alpha_exp_coef[2*(h+H*t)] = last_exp_coef1;
            alpha_exp_coef[1+2*(h+H*t)] = last_exp_coef2;
        }
    }
    plhs[0] = mxCreateDoubleScalar(max_last_exp_coef);
    plhs[1] = mxCreateDoubleScalar(last_exp_coef1);
    plhs[2] = mxCreateDoubleScalar(last_exp_coef2);
    
    return;   
}
