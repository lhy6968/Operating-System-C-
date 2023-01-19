#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <signal.h>

int main(int argc,char *argv[]){

	/* fork a child process */
	printf("Process start to fork\n");
	int status;
    	pid_t pid = fork();

    	if (pid < 0) {
        	printf ("Fork error!\n");
    	}
    	else {
	//child process
        	if (pid == 0) {
            
            	int i;
            	char *arg[argc];
            
            	for(i=0;i<argc-1;i++){
                	arg[i]=argv[i+1];
            	}
            	arg[argc-1]=NULL;
	printf("I'm the parent process,my pid = %d\n", getppid());
	printf("I'm the child process,my pid = %d\n", getpid());
            	printf("Child process start to execute the program:\n");
	/* execute test program */ 
            	execve(arg[0],arg,NULL);
            
            	printf("Continue to run original child process!\n");
            
            	perror("execve");
            	exit(EXIT_FAILURE);
        }   
        	//parent process
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
		printf("child proccess get SIGHUP signal\n");
		printf("child process is hang up\n");
		break;
		case 2 :
		printf("child proccess get SIGINT signal\n");
		printf("child process is interrupted\n");
		printf("interrupt\n");
		break;
		case 3 :
		printf("child proccess get SIGQUIT signal\n");
		printf("child process is quited and terminated\n");
		break;
		case 4 :
		printf("child proccess get SIGILL signal\n");
		printf("child process occurs an illegal error\n");
		break;
		case 5 :
		printf("child proccess get SIGTRAP signal\n");
		printf("child process reach a breakpoint\n");
		break;
		case 6 :
		printf("child proccess get SIGABORT signal\n");
		printf("child process is abort by abort signal\n");
		break;
		case 7 :
		printf("child proccess get SIGBUS signal\n");
		printf("child process is abort by bus signal\n");
		break;
		case 8 :
		printf("child proccess get SIGFPE signal\n");
		printf("child process has a SIGFPE error\n");
		break;
		case 9 :
		printf("child proccess get SIGKILL signal\n");
		printf("child process is killed and terminated\n");
		break;
		case 11 :
		printf("child proccess get SIGSEGV signal\n");
		printf("child process occurs memory access violation\n");
		break;
		case 13 :
		printf("child proccess get SIGPIPE signal\n");
		printf("child process is piped and terminated\n");
		break;
		case 14 :
		printf("child proccess get SIGALRM signal\n");
		printf("child process is alarm by alarm signal\n");
		break;
		case 15 :
		printf("child proccess get SIGTERM signal\n");
		printf("child process is abnormally and forcefully terminated\n");
		break;		
		default:
		break;
	}

	printf("CHILD EXCUTION FAILED!\n");

            }
            else if(WIFSTOPPED(status)){
		printf("Child proccess get SIGSTOP signal\n");
		printf("child process stopped\n");
                	printf("CHILD PROCESS STOPPED\n");
            }
            else{
                printf("CHILD PROCESS CONTINUED\n");
            }
            exit(1);
        }
    }
    return 0;
}
