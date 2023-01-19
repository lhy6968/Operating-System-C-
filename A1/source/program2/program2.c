#include <linux/module.h>
#include <linux/sched.h>
#include <linux/pid.h>
#include <linux/kthread.h>
#include <linux/kernel.h>
#include <linux/err.h>
#include <linux/slab.h>
#include <linux/printk.h>
#include <linux/jiffies.h>
#include <linux/kmod.h>
#include <linux/fs.h>


MODULE_LICENSE("GPL");

static struct task_struct *task;

struct wait_opts { enum pid_type wo_type;
	int wo_flags;
	struct pid *wo_pid;
	struct siginfo __user *wo_info;
	long __user *wo_stat;
	struct rusage __user *wo_rusage;
	wait_queue_t child_wait;
	int notask_error;
};

extern long do_wait(struct wait_opts *wo);

extern long _do_fork(unsigned long clone_flags,
	      unsigned long stack_start,
	      unsigned long stack_size,
	      int __user *parent_tidptr,
	      int __user *child_tidptr,
	      unsigned long tls);

extern int do_execve(struct filename *filename,
	const char __user *const __user *__argv,
	const char __user *const __user *__envp);

extern struct filename * getname(const char __user * filename);

int my_exec(void){
	int result;
	const char path[] = "/home/seed/work/assignment1/source/program2/test";
	const char *const argv[]={path,NULL,NULL};
	const char *const envp[]={"HOME=/","PATH=/sbin:/user/sbin:/bin:/usr/bin",NULL};
	struct filename * my_filename = getname(path);
	printk("child process\n");
	result=do_execve(my_filename,argv,envp);

	if(!result)return 0;

	do_exit(result);
}

void my_wait(pid_t pid){
	int a;
	int status;
	struct wait_opts wo;
	struct pid *wo_pid=NULL;
	enum pid_type type;
	type=PIDTYPE_PID;
	wo_pid=find_get_pid(pid);

	wo.wo_type = type;
	wo.wo_pid = wo_pid;
	wo.wo_flags = WEXITED | WUNTRACED;
	wo.wo_info = NULL;
	wo.wo_stat = (long __user*)&status;
	wo.wo_rusage = NULL;
	a = do_wait(&wo);
	if (*wo.wo_stat == 1){
		printk("get SIGHUP signal\n");
		printk("child process is hang up\n");
	}
	else if (*wo.wo_stat == 2){
		printk("get SIGINT signal\n");
		printk("child process is interrupted\n");
	}
	else if (*wo.wo_stat == 3){
		printk("get SIGQUIT signal\n");
		printk("child process is quited and terminated\n");
	}
	else if (*wo.wo_stat == 4){
		printk("get SIGILL signal\n");
		printk("child process occurs an illegal error\n");
	}
	else if (*wo.wo_stat == 5){
		printk("get SIGTRAP signal\n");
		printk("child process reach a breakpoint\n");
	}
	else if (*wo.wo_stat == 6){
		printk("get SIGABORT signal\n");
		printk("child process is abort by abort signal\n");
	}
	else if (*wo.wo_stat == 7){
		printk("get SIGBUS signal\n");
		printk("child process has a bus error\n");
	}
	else if (*wo.wo_stat == 8){
		printk("get SIGFPE signal\n");
		printk("child process is abort by SIGFPE signal\n");
	}
	else if (*wo.wo_stat == 9){
		printk("get SIGKILL signal\n");
		printk("child process is killed and terminated\n");
	}
	else if (*wo.wo_stat == 11){
		printk("get SIGSEGV signal\n");
		printk("child process occurs memory access violation\n");
	}
	else if (*wo.wo_stat == 13){
		printk("get SIGPIPE signal\n");
		printk("child process is piped and terminated\n");
	}
	else if (*wo.wo_stat == 14){
		printk("get SIGALRM signal\n");
		printk("child process is alarm by alarm signal\n");
	}
	else if (*wo.wo_stat == 15){
		printk("get SIGTERM signal\n");
		printk("child process is abnormally and forcefully terminated\n");
	}
	else if (*wo.wo_stat == 4991){
		printk("get SIGSTOP signal\n");
		printk("child process stopped\n");
	}
	else{
		printk("child process normally\n");
	}
	printk("The return signal is %ld\n", *wo.wo_stat);
	put_pid(wo_pid);
}

int my_fork(void *argc){	
	pid_t pid;
	int i;
	struct k_sigaction *k_action = &current->sighand->action[0];
	for(i=0;i<_NSIG;i++){
		k_action->sa.sa_handler = SIG_DFL;
		k_action->sa.sa_flags = 0;
		k_action->sa.sa_restorer = NULL;
		sigemptyset(&k_action->sa.sa_mask);
		k_action++;
	}
	/* fork a process using do_fork */
	pid = _do_fork(SIGCHLD, (unsigned long)&my_exec,0,NULL,NULL,0);
	/* execute a test program in child process */
	printk("The child process has pid = %d\n",pid);
	printk("This is the parent process,pid = %d\n",(int)current->pid);
	
	/* wait until child process terminates */
	my_wait(pid);

	return 0;
}

static int __init program2_init(void){

	printk("module_init\n");
	
	/* write your code here */
	printk("module_init create kthread start\n");
	/* create a kernel thread to run my_fork */
	
	task=kthread_create(&my_fork,NULL,"MyThread");

	if(!IS_ERR(task)){
		printk("module_init kthread starts\n");
		wake_up_process(task);
	}
	return 0;

}

static void __exit program2_exit(void){
	printk("module_exit\n");
}

module_init(program2_init);
module_exit(program2_exit);
