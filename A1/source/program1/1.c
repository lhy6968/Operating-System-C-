#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <signal.h>

int main(int argc, char *argv[]){
    /* fork a child process */
    pid_t pid;
    int status;
    printf("Process start to fork.\n");
    pid = fork();

    /* execute test program */ 
    if (pid==-1){
	perror("fork");
    	exit(1);
    }
    else{
	//the child process
	if (pid==0){
	    int i;
	    char *arg[argc];
	    for(i=0;i<argc-1;i++){
                arg[i]=argv[i+1];
            }
            arg[argc-1]=NULL;
    	    printf("I'm the Parent process,my pid = %d\n", getppid());
	    printf("I'm the Child process,my pid = %d\n", getpid());
	    printf("Child process start to execute the program\n");
	    execve(arg[0],arg,NULL);
	    exit(0);

	}
	//the parent process
	else{
            waitpid(pid, &status, WUNTRACED);
	    printf("Parent process revieving the SIGCHLD signal.\n");
	    if(WIFEXITED(status)){
		printf("Normal termination with EXIT STATUS = %d\n",WEXITSTATUS(status));	    
	    }
	    else if(WIFSIGNALED(status)){
		switch (WTERMSIG(status))
		    {
				case 1 :
					printf("Child proccess get SIGHUP signal\n");
					printf("Child proccess is abort by hangup signal\n");
					break;
				case 2 :
					printf("Child proccess get SIGINT signal\n");
					printf("Child proccess is abort by interrupt signal\n");
					printf("interrupt\n");
					break;
				case 3 :
					printf("Child proccess get SIGQUIT signal\n");
					printf("Child proccess is abort by quit signal\n");
					break;
				case 4 :
					printf("Child proccess get SIGILL signal\n");
					printf("Child proccess is abort by illegal_instr signal\n");
					break;
				case 5 :
					printf("Child proccess get SIGTRAP signal\n");
					printf("Child proccess is abort by trap signal\n");
					break;
				case 6 :
					printf("Child proccess get SIGABORT signal\n");
					printf("Child proccess is abort by abort signal\n");
					break;
				case 7 :
					printf("Child proccess get SIGBUS signal\n");
					printf("Child proccess is abort by bus signal\n");
					break;
				case 8 :
					printf("Child proccess get SIGFPE signal\n");
					printf("Child proccess is abort by float signal\n");
					break;
				case 9 :
					printf("Child proccess get SIGKILL signal\n");
					printf("Child proccess is abort by kill signal\n");
					break;
				case 11 :
					printf("Child proccess get SIGSEGV signal\n");
					printf("Child proccess is abort by segment_fault signal\n");
					break;
				case 13 :
					printf("Child proccess get SIGPIPE signal\n");
					printf("Child proccess is abort by pipe signal\n");
					break;
				case 14 :
					printf("Child proccess get SIGALRM signal\n");
					printf("Child proccess is abort by alarm signal\n");
					break;
				case 15 :
					printf("Child proccess get SIGTERM signal\n");
					printf("Child proccess is abort by terminate signal\n");
					break;
				
				default:
					break;
				}
		printf("CHILD EXECUTION FAILED!!\n");
	    }
	else if(WIFSTOPPED(status)){
		printf("child process get SIGSTOP signal\n");
		printf("child process stopped\n");
		printf("CHILD PROCESS STOPPED\n");
	}
	else{
		printf("CHILD PROCESS CONTINUED\n");
	};
	    exit(1);

	};

     };
	return 0;

}