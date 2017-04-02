
/* calculate xi, in forward_bacward_hidden_unit_optimized.
 * syntax:         vmid_out(T, H, 2, 2) = calculate_psi(alpha, alpha_exp_coef, beta, beta_exp_coef, psi, log_h_mid_alpha, T, H)
 *
 */

#include <string.h>
#include <math.h>
#include "mex.h"
#include "matrix.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
    /* Check inputs */
    if(nrhs != 8)
        mexErrMsgTxt("Function should have four inputs.");
    
    /* Get inputs */
    
    double* alpha = mxGetPr(prhs[0]);
    double* alpha_exp_coef = mxGetPr(prhs[1]);
    double* beta = mxGetPr(prhs[2]);
    double* beta_exp_coef = mxGetPr(prhs[3]);
    double* psi = mxGetPr(prhs[4]);
    double* h_mid_alpha = mxGetPr(prhs[5]);
    int T =  (int)*mxGetPr(prhs[6]);
    int H =  (int)*mxGetPr(prhs[7]);
//     mexPrintf("T = %d, H = %d  \n", T, H);
    
    int N[4];
    N[0] = T;
    N[1] = H;
    N[2] = 2;
    N[3] = 2;
    plhs[0] = mxCreateNumericArray(4, N, mxDOUBLE_CLASS, mxREAL);
    double* temp = mxGetPr(mxCreateNumericArray(4, N, mxDOUBLE_CLASS, mxREAL));
    double* vmid_out = mxGetPr(plhs[0]);
    
    double v_sum = 0;
    for(int h=0; h<H; h++)
    {
        v_sum += h_mid_alpha[h];
    }
    
    for (int t=0; t<T; t++)
        for(int h=0; h<H; h++)
            for(int i=0; i<2; i++)
                for(int j=0; j<2; j++)
                { 
                    int index = t + T*H*2*j + T*H*i + T*h;
                    if(t<T-1)
                        temp[index] = alpha[i+2*(h+H*t)] + alpha_exp_coef[i+2*(h+H*t)] + psi[index+1] + 
                                beta[j+2*(h+H*(t+1))] + beta_exp_coef[j+2*(h+H*(t+1))];
                    else
                        temp[index] = beta[j+2*h] + beta_exp_coef[j+2*h];
                    vmid_out[index] = temp[index] + v_sum - h_mid_alpha[h];
                }

    
    return;
}
