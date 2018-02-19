// nvcc 014 sgemv .c -lcublas
# include < stdio .h >
# include < stdlib .h >
# include < math .h >
# include < cuda_runtime .h >
# include " cublas_v2 .h"
# define IDX2C (i ,j , ld ) ((( j )*( ld ))+( i ))
# define m 6 // number of rows of a
# define n 5 // number of columns of a


__global__ void execute_gemv(cublasHandle_t handle, float * d_a, float * d_x, float * d_y){
    float al = 1.0;
    float beta = 0.0;
    stat=cublasSgemv(handle,CUBLAS_OP_N,m,n,&al,d_a,m,d_x,1,&beta,d_y,1);
}

int main ( void ){
    cudaError_t cudaStat ; // cudaMalloc status
    cublasStatus_t stat ; // CUBLAS functions status
    cublasHandle_t handle ; // CUBLAS context

    int i,j; // i-row index , j- column index
    float * a; // a -mxn matrix on the host
    float * x; // x - n- vector on the host
    float * y; // y - m- vector on the host

    a=( float *) malloc (m*n* sizeof ( float )); // host mem. alloc for a
    x=( float *) malloc (n* sizeof ( float )); // host mem. alloc for x
    y=( float *) malloc (m* sizeof ( float )); // host mem. alloc for y

    // define an mxn matrix a - column by column
    int ind =11; // a:
    for(j=0;j<n;j ++){
        for(i=0;i<m;i ++){
            a[ IDX2C (i,j,m )]=( float )ind ++; // 13 ,19 ,25 ,31 ,37
        } 
    }

    printf ("a:\n");

    for (i=0;i<m;i ++){
        for (j=0;j<n;j ++){
            printf (" %4.0 f",a[ IDX2C (i,j,m )]); // print a row by row
        }
        printf ("\n");
    }

    for(i=0;i<n;i++) x[i ]=1.0 f; // x={1 ,1 ,1 ,1 ,1}^T
    for(i=0;i<m;i++) y[i ]=0.0 f; // y={0 ,0 ,0 ,0 ,0 ,0}^T

    // on the device
    float * d_a; // d_a - a on the device
    float * d_x; // d_x - x on the device
    float * d_y; // d_y - y on the device

    cudaStat = cudaMalloc (( void **)& d_a ,m*n* sizeof (*a)); // device
    // memory alloc for a
    cudaStat = cudaMalloc (( void **)& d_x ,n* sizeof (*x)); // device
    // memory alloc for x
    cudaStat = cudaMalloc (( void **)& d_y ,m* sizeof (*y)); // device
    // memory alloc for y
    stat = cublasCreate (& handle );
    stat = cublasSetMatrix (m,n, sizeof (*a) ,a,m,d_a ,m); // cp a- >d_a
    stat = cublasSetVector (n, sizeof (*x) ,x ,1 ,d_x ,1); // cp x- >d_x
    stat = cublasSetVector (m, sizeof (*y) ,y ,1 ,d_y ,1); // cp y- >d_y
    float al =1.0 f; // al =1
    float bet =0.0 f; // bet =0
    // matrix - vector multiplication : d_y = al*d_a *d_x + bet *d_y
    // d_a - mxn matrix ; d_x - n-vector , d_y - m- vector ;
    // al ,bet - scalars

    execute_gemv <<<1,1>>> (handle, d_a, d_x, d_y);

    stat = cublasGetVector (m, sizeof (*y) ,d_y ,1 ,y ,1); // copy d_y - >y
    printf ("y after Sgemv ::\ n");
    for(j=0;j<m;j ++){
        printf (" %5.0 f",y[j]); // print y after Sgemv
        printf ("\n");
    }

    cudaFree (d_a ); // free device memory
    cudaFree (d_x ); // free device memory
    cudaFree (d_y ); // free device memory
    cublasDestroy ( handle ); // destroy CUBLAS context
    free (a); // free host memory
    free (x); // free host memory
    free (y); // free host memory
    return EXIT_SUCCESS ;
}
