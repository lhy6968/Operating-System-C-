#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <curses.h>
#include <termios.h>
#include <fcntl.h>

#define ROW 10
#define COLUMN 50
#define NUM_THREADS 10
pthread_cond_t threshold;
pthread_mutex_t mutex;

struct Node{
	int x , y; 
	Node( int _x , int _y ) : x( _x ) , y( _y ) {}; 
	Node(){} ; 
} frog ; 
int status=-999;
int thread_ids[10] = {1,2,3,4,5,6,7,8,9,10};

char map[ROW+10][COLUMN] ; 

// Determine a keyboard is hit or not. If yes, return 1. If not, return 0. 
int kbhit(void){
	struct termios oldt, newt;
	int ch;
	int oldf;

	tcgetattr(STDIN_FILENO, &oldt);

	newt = oldt;
	newt.c_lflag &= ~(ICANON | ECHO);

	tcsetattr(STDIN_FILENO, TCSANOW, &newt);
	oldf = fcntl(STDIN_FILENO, F_GETFL, 0);

	fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

	ch = getchar();

	tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
	fcntl(STDIN_FILENO, F_SETFL, oldf);

	if(ch != EOF)
	{
		ungetc(ch, stdin);
		return 1;
	}
	return 0;
}


void *logs_move( void *t ){
	int *index = (int*) t;
	int length = NUM_THREADS-1;
	int num = 20;
	int arr[length];
	int speed[length];
	char label = '|';
	for (int i = 0; i<length;i++){
		speed[i]=rand()%(1000000-500000+1)+500000;
	}
	for (int i = 0; i<length;i++){
		arr[i] = rand()%(48-0+1)+0;
	}
	while (true){
		pthread_mutex_lock(&mutex);	
		for(int i = 0; i < COLUMN-1; ++i){
			map[*index][i]=' ';
			}
		for (int i = 0; i< COLUMN-1; i++){
			map[ROW][i]='|';
		}
		if (*index!=10){
			if (*index%2==0){
				for (int i = 0; i < num; i++){
					map[*index][(arr[*index-1]+i)%49] = '=';
				}
				if (frog.x>0 && frog.x<ROW && frog.x==*index){
					frog.y += 1;
				}
				for (int i = 0; i<length;i++){
					arr[i] += 1;
				}
			}
			else{
				for (int i = 0;i < num;i++){
					map[*index][(49+(arr[*index-1]-i)%49)%49] = '=';	
				}
				if (frog.x>0 && frog.x<ROW && frog.x==*index){
					frog.y -= 1;
				}
				for (int i = 0; i<length;i++){
					arr[i] -= 1;
				}
			}
		}
		else{
			map[frog.x][frog.y]='0';
			/*  Check keyboard hits, to change frog's position or quit the game. */
			if (kbhit()){
		 		char dir = getchar();
		 		if (dir == 'w' || dir == 'W'){
		 			frog.x-=1;
					if (map[frog.x][frog.y] == ' ') status = 0;
					if ((frog.y<=0 || frog.y>=COLUMN-2)&frog.x!=ROW) status = 0;
					if (frog.x == 0) status=1;
					map[frog.x+1][frog.y]=label;
					label = map[frog.x][frog.y];
					map[frog.x][frog.y]='0';
		 		}else if(dir == 'a' || dir == 'A'){
					if(!(frog.x==ROW&frog.y==0)){
						frog.y-=1;
						if (map[frog.x][frog.y] == ' ') status = 0;
						if ((frog.y<=0 || frog.y>=COLUMN-2)&frog.x!=ROW) status = 0;
						if (frog.x == 0) status=1;
						map[frog.x][frog.y+1]=label;
						label = map[frog.x][frog.y];
						map[frog.x][frog.y]='0';
					}
		 		}else if (dir == 's' || dir == 'S'){
					if (frog.x<ROW){
						frog.x+=1;
						if (map[frog.x][frog.y] == ' ') status = 0;
						if ((frog.y<=0 || frog.y>=COLUMN-2)&frog.x!=ROW) status = 0;
						if (frog.x == 0) status=1;
						map[frog.x-1][frog.y]=label;
						label = map[frog.x][frog.y];
						map[frog.x][frog.y]='0';
					}
				}else if(dir == 'd' || dir == 'D'){
					if(!(frog.x==ROW&frog.y==COLUMN-2)){
						frog.y+=1;
						if (map[frog.x][frog.y] == ' ') status = 0;
						if ((frog.y<=0 || frog.y>=COLUMN-2)&frog.x!=ROW) status = 0;
						if (frog.x == 0) status=1;
						map[frog.x][frog.y-1]=label;
						label = map[frog.x][frog.y];
						map[frog.x][frog.y]='0';
					}
				}else if(dir == 'q' || dir == 'Q'){
					map[frog.x][frog.y]='0';
					status = -1;
				}
			}
			else{
				map[frog.x][frog.y]='0';
				if (map[frog.x][frog.y] == ' ') status = 0;
				if ((frog.y<=0 || frog.y>=COLUMN-2)&frog.x!=ROW) status = 0;
				if (frog.x == 0) status=1;
			}
			/*  Print the map on the screen  */
			printf("\033[0;0H\033[2J");
			for(int i = 0; i <= ROW; ++i){
				puts( map[i]);
			}
		}
		pthread_mutex_unlock(&mutex);
		/*  Check game's status  */
		if (status==-1 || status == 0 || status ==1){
			printf("\033[0;0H\033[2J");
			pthread_exit(NULL);
		}
		if (*index == 10){
			usleep(100000);
		}
		else{
			usleep(speed[*index-1]);
		}	
	}
}


int main( int argc, char *argv[] ){

	// Initialize the river map and frog's starting position
	memset( map , 0, sizeof( map ) ) ;
	int i , j ; 
	for( i = 1; i < ROW; ++i ){	
		for( j = 0; j < COLUMN - 1; ++j )	
			map[i][j] = ' ' ;  
	}	

	for( j = 0; j < COLUMN - 1; ++j )	
		map[ROW][j] = map[0][j] = '|' ;

	for( j = 0; j < COLUMN - 1; ++j )	
		map[0][j] = map[0][j] = '|' ;

	frog = Node( ROW, (COLUMN-1) / 2 ) ; 
	map[frog.x][frog.y] = '0' ; 
	//Print the map into screen
	for( i = 0; i <= ROW; ++i)	
		puts( map[i] );

	/*  Create pthreads for wood move and frog control.  */
	pthread_t threads[NUM_THREADS];	
	pthread_mutex_init(&mutex,NULL);
	pthread_cond_init(&threshold,NULL);
	for (int i = 0; i <NUM_THREADS; i++){
		pthread_create(&threads[i],NULL,logs_move,(void *)&thread_ids[i]);
	}
	for (int i = 0; i <NUM_THREADS; i++){
		pthread_join(threads[i],NULL);
	}
	/*  Display the output for user: win, lose or quit.  */
	if (status==-1){
		printf("You exit the game\n");
	}else if (status==1){
		printf("You win the game\n");
	}else if (status==0){
		printf("You lose the game\n");
	}
	pthread_mutex_destroy(&mutex);
	pthread_cond_destroy(&threshold);
	pthread_exit(NULL);

	return 0;

}
