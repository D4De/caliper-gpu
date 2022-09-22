#define ROWS 5
#define COLS 5

// Electro-Migration related parameters
#define BETA 2
#define ACTIVATIONENERGY 0.48
#define BOLTZMANCONSTANT 8.6173324*0.00001
#define CONST_JMJCRIT 1500000
#define CONST_N 1.1
#define CONST_ERRF 0.88623
#define CONST_A0 30000 //cross section = 1um^2  material constant = 3*10^13

// Thermal model parameters
#define ENV_TEMP 295 //room temperature
#define SELF_TEMP 40 //self contribution
#define NEIGH_TEMP 5 //neighbor contribution

#define getAlpha(temp) ((CONST_A0 * (pow(CONST_JMJCRIT,(-CONST_N))) * exp(ACTIVATIONENERGY / (BOLTZMANCONSTANT * temp))) / CONST_ERRF)

#define NTESTS 100000
#define BETA 2
#define MIN_NUM_OF_TRIALS 30

#define RANDOMSEED_STR "RANDOM"

// Support function to allow arbitrary confidence intervals
#define INV_ERF_ACCURACY 10e-6

void tempModel(double loads[][COLS], double temps[][COLS]);

void tempModel(double loads[][COLS], double temps[][COLS], int rows, int cols) {
    double temp;
    int i, j, k, h;
    for (i = 0; i < rows; i++)
        for (j = 0; j < cols; j++) {
            for (k = -1, temp = 0; k < 2; k++)
                for (h = -1; h < 2; h++)
                    if ((k != 0 || h != 0) && k != h && k != -h && i + k >= 0 && i + k < rows && j + h >= 0 && j + h < cols){
                        temp += loads[i + k][j + h] * NEIGH_TEMP;
                    }
            temps[i][j] = ENV_TEMP + loads[i][j] * SELF_TEMP + temp;
        }
}