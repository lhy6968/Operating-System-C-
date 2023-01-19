#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <wait.h>
#include <unistd.h>
#include<string.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#define MAX_LINE 1024
extern char *strcat(char *dest, const char *src);
extern size_t strlen(const char *str);

int main(int argc,char *argv[]){

	//clear all the contents in the tmp.txt file
	int tmp = open("tmp.txt", O_WRONLY | O_TRUNC);
	close(tmp);

	pid_t pid;
	int status;
	int zero_pid;
	int i = 1;
    	for(i; i < argc; i = i + 1) {
         		zero_pid = pid;		 
		pid = fork();	 
		if(pid > 0) {
             			break;
         		}else{
			continue;		
		}	 		 
  	}
	//record all the pid into the tmp.txt
	int tmp_2 = open("tmp.txt",O_RDWR|O_CREAT|O_APPEND,0666);	
	char buf[999];
	sprintf(buf, "%d\n", getpid());
	write(tmp_2, buf, strlen(buf));	
	
	if (pid > 0){
		wait(&status);
	}	
	close(tmp_2);

	if (zero_pid == 0){
		char *arg[argc];	
		for(int j=0;j<=argc-i;j=j+1){
			arg[j]=argv[j+i-1];
		}
		for(int k = 1; k < i; k=k+1){
			arg[argc-k]=NULL;
		}
		execve(arg[0],arg,NULL);
	}
	char buf2[MAX_LINE];
 	FILE *fp;
 	int len;
 	if((fp = fopen("tmp.txt","r")) == NULL)
 	{
 	perror("fail to read");
 	exit (1) ;
 	}
	printf("the process tree:");
	int num = 0;
	while(fgets(buf2,MAX_LINE,fp) != NULL)
 	{
 	len = strlen(buf2);
 	buf2[len-1] = '\0';
	num += 1;
	if (num<argc){
		printf("%s->",buf2);
	}
	else{
		printf("%s",buf2);
	}
	
 	}
	printf("\n");
	
	
    	return 0;
}
