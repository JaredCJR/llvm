#include <stdlib.h>
#include <string.h>
#include <stdio.h>

int main(int argc, char* argv[]) 
{
    char CmdBuffer[4096] = {};
    char space[] = " ";
    char postfix[] = ".py";
    strncat(CmdBuffer, argv[0], strlen(argv[0]));
    strncat(CmdBuffer, postfix, strlen(postfix));
    for(int i = 1;i < argc; i++) {
        strncat(CmdBuffer, space, strlen(space));
        strncat(CmdBuffer, argv[i], strlen(argv[i]));
    }
    system(CmdBuffer);
    return 0;
}
