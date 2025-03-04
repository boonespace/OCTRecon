#include "OCTRecon.h"

int main(){

    // Adjust according to the actual project directory structure
    DataStorage ds;
    ds.getFromFolder("../../test_data", ".bin");

    // Perform image reconstruction
    Recon recon(2056, 280, 280, DataType::UINT16);
    recon.readData(ds.readname(0));
    recon.reconstruction();

    return 0;
}