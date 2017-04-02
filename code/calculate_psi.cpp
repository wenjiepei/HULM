
/* calculate the psi, correponding to the fucntion 'calculate_psi_using_mid_result' in forward_bacward_hidden_unit_optimized.
 * syntax:         psi(:, :, :, :) = calculate_psi(model.V * y, mid_psi_result(:, :, :, :), T, H)
 *
 */

#include <string.h>
#include <math.h>
#include "mex.h"
#include "matrix.h"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
    /* Check inputs */
    if(nrhs != 4)
        mexErrMsgTxt("Function should have four inputs.");
    
    /* Get inputs */
    const mxArray* model_V_psi = prhs[0];
    const mxArray* mid_psi_result = prhs[1];
    
    double* mid_psi = mxGetPr(mid_psi_result);
    double* V_psi = mxGetPr(model_V_psi);
    int T =  (int)*mxGetPr(prhs[2]);
    int H =  (int)*mxGetPr(prhs[3]);
//     mexPrintf("T = %d, H = %d  \n", T, H);
    
    int N[4];
    N[0] = T;
    N[1] = H;
    N[2] = 2;
    N[3] = 2;
    plhs[0] = mxCreateNumericArray(4, N, mxDOUBLE_CLASS, mxREAL);
    double * psi_v = mxGetPr(plhs[0]);
    
    for (int t=0; t<T; t++)
        for(int h=0; h<H; h++)
            for(int i=0; i<2; i++)
            {
                int index0 = t + T*H*i + T*h;
                int index1 = t + T*H*2 + T*H*i + T*h;
                psi_v[index0] = mid_psi[index0];
                psi_v[index1] = mid_psi[index1] + V_psi[h];
                
            }
    
    
    return;
}
