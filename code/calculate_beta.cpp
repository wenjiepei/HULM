
/* calculate beta from t=2 to T, in forward_bacward_hidden_unit_optimized.
 * syntax:       [last_exp_coef, last_exp_coef1, last_exp_coef2] = calculate_beta(beta, beta_exp_coef, psi, last_exp_coef1, last_exp_coef2, T, H)
 * since we do not use const pointer, hence the value of beta and beta_exp_coef is updated automated. 
 *
 */

#include <math.h>
#include "mex.h"
#include "matrix.h"

double get_max_beta_at_t(double* beta, int t, int T, int H)
{
    double rtn = beta[2*(0+H*t)];
    
    for(int i=1; i<H; i++)
    {
        if(rtn < beta[2*(i+H*t)])
            rtn = beta[2*(i+H*t)];
    }
    
    return rtn;
}

double get_max_psi_at_t(double* psi, double* beta, int t, int index1, int index2, int T, int H)
{
    double rtn = psi[t+T*( 0+ H*(index1 + 2*index2) )] + beta[1+2*(0+H*t)];
    
    for(int i=1; i<H; i++)
    {
        if(rtn < psi[t+T*( i+ H*(index1 + 2*index2) )] + beta[1+2*(i+H*t)])
            rtn = psi[t+T*( i+ H*(index1 + 2*index2) )] + beta[1+2*(i+H*t)];
    }
    
    return rtn;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
    /* Check inputs */
    if(nrhs != 7)
        mexErrMsgTxt("Function should have seven inputs.");
    
    /* get inputs */
    double* beta_initial = mxGetPr(prhs[0]);
    double* beta_exp_coef_initial = mxGetPr(prhs[1]);
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
    plhs[0] = mxCreateNumericArray(3, N, mxDOUBLE_CLASS, mxREAL);
    plhs[0] = mxDuplicateArray(prhs[0]);
    double* beta = mxGetPr(plhs[0]);
    plhs[1] = mxCreateNumericArray(3, N, mxDOUBLE_CLASS, mxREAL);
    plhs[1] = mxDuplicateArray(prhs[1]);
    double* beta_exp_coef = mxGetPr(plhs[1]);
    
    /* calculate beta by performing backward pass */
    double pa1, pa2, max_pa;

    for(int t=T-2; t>-1; t--)
    {
        double max_beta = get_max_beta_at_t(beta, t+1, T, H);
        double max_temp_psi1 = get_max_psi_at_t(psi, beta, t+1, 0, 1, T, H);
        double max_temp_psi2 = get_max_psi_at_t(psi, beta, t+1, 1, 1, T, H);
        double exp_coef1 = (max_beta + last_exp_coef1 > max_temp_psi1 + last_exp_coef2) ? (max_beta + last_exp_coef1) : (max_temp_psi1 + last_exp_coef2);
        double exp_coef2 = (max_beta + last_exp_coef1 > max_temp_psi2 + last_exp_coef2) ? (max_beta + last_exp_coef1) : (max_temp_psi2 + last_exp_coef2);
        
        for(int h=0; h<H; h++)
        {
            pa1 = beta[2*(h+H*(t+1))] + last_exp_coef1 - exp_coef1;
            pa2 = beta[1+2*(h+H*(t+1))] + last_exp_coef2 + psi[t+1+T*(h+H*(2*1))] - exp_coef1;
//             beta[2*(h+H*t)] = log(exp(pa1) + exp(pa2)); 
            max_pa = (pa1 > pa2) ? pa1 : pa2;
            beta[2*(h+H*t)] = max_pa + log(exp(pa1-max_pa) + exp(pa2-max_pa));
            
            pa1 = beta[2*(h+H*(t+1))] + last_exp_coef1 - exp_coef2;
            pa2 = beta[1+2*(h+H*(t+1))] + last_exp_coef2 + psi[t+1+T*(h+H*(1+2*1))] - exp_coef2;
//             beta[1+2*(h+H*t)] = log(exp(pa1) + exp(pa2));            
            max_pa = (pa1 > pa2) ? pa1 : pa2;
            beta[1+2*(h+H*t)] = max_pa + log(exp(pa1-max_pa) + exp(pa2-max_pa));
        }
        last_exp_coef1 = exp_coef1;
        last_exp_coef2 = exp_coef2;
        
        for(int h=0; h<H; h++)
        {
            beta_exp_coef[2*(h+H*t)] = last_exp_coef1;
            beta_exp_coef[1+2*(h+H*t)] = last_exp_coef2;
        }
    }
        
    return;

}
