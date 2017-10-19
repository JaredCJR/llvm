#include <stdlib.h>
#include <string>
#include <stdio.h>

int main(int argc, char* argv[]) 
{
    char CmdBuffer[4096] = {};
    std::string cmd = argv[0];
    cmd += ".py";
    std::string retFileName = "./ReturnValue";
    system(("rm -f " + retFileName).c_str());

    for(int i = 1;i < argc; i++) {
        cmd += " ";
        cmd += argv[i];
    }
    system(cmd.c_str());
    FILE *pFile = fopen( retFileName.c_str(), "r");
    if(pFile) {
        int ret;
        fread(&ret, sizeof(ret), 1, pFile);
        return ret - '0';
    }else {
        return 87;
    }
    return 0;
}
