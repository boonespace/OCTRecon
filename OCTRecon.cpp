#include "OCTRecon.h"

int main()
{
    // Adjust according to the actual project directory structure
    DataStorage ds;
    ds.getFromFolder("../../test_data", ".bin");

    // Perform image reconstruction
    Recon recon(2056, 280, 280, DataType::UINT16);

    for (int i = 0; i < ds.length; i++)
    {
        recon.readData(ds.readname(i));
        recon.reconstruction();
    }

    return 0;
}