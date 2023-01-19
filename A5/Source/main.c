#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/stat.h>
#include <linux/fs.h>
#include <linux/workqueue.h>
#include <linux/sched.h>
#include <linux/interrupt.h>
#include <linux/slab.h>
#include <linux/cdev.h>
#include <linux/delay.h>
#include <asm/uaccess.h>
#include "ioc_hw5.h"

MODULE_LICENSE("GPL");

#define PREFIX_TITLE "OS_AS5"

//this part is used to define the contents in DMA
// DMA
#define DMA_BUFSIZE 64
#define DMASTUIDADDR 0x0        // Student ID
#define DMARWOKADDR 0x4         // RW function complete
#define DMAIOCOKADDR 0x8        // ioctl function complete
#define DMAIRQOKADDR 0xc        // ISR function complete
#define DMACOUNTADDR 0x10       // interrupt count function complete
#define DMAANSADDR 0x14         // Computation answer
#define DMAREADABLEADDR 0x18    // READABLE variable for synchronize
#define DMABLOCKADDR 0x1c       // Blocking or non-blocking IO
#define DMAOPCODEADDR 0x20      // data.a opcode
#define DMAOPERANDBADDR 0x21    // data.b operand1
#define DMAOPERANDCADDR 0x25    // data.c operand2
void *dma_buf;

//this part is the information of the devide in detail
//DEVICE
#define DEV_NAME "mydev"        // name for alloc_chrdev_region
#define DEV_BASEMINOR 0         // baseminor for alloc_chrdev_region
#define DEV_COUNT 1             // count for alloc_chrdev_region
static int dev_major;
static int dev_minor;
static struct cdev *dev_cdev;

//this part is to define the readable and unreadable flag in DMA
#define UNREADABLE 0
#define READABLE 1

//define IRQ_NUM 
#define IRQ_NUM 1

//count the number of interrupt
int interrupt_num;

// Declaration for file operations
static ssize_t drv_read(struct file *filp, char __user *buffer, size_t, loff_t*);
static int drv_open(struct inode*, struct file*);
static ssize_t drv_write(struct file *filp, const char __user *buffer, size_t, loff_t*);
static int drv_release(struct inode*, struct file*);
static long drv_ioctl(struct file *, unsigned int , unsigned long );

//connect the user operations with the device operations 
// cdev file_operations
static struct file_operations fops = {
      owner: THIS_MODULE,
      read: drv_read,
      write: drv_write,
      unlocked_ioctl: drv_ioctl,
      open: drv_open,
      release: drv_release,
};

//use my student id to define the device id
int dev_id = 118010138;

// in and out function
void myoutc(unsigned char data,unsigned short int port);
void myouts(unsigned short data,unsigned short int port);
void myouti(unsigned int data,unsigned short int port);
unsigned char myinc(unsigned short int port);
unsigned short myins(unsigned short int port);
unsigned int myini(unsigned short int port);

// Work routine
static struct work_struct *work;

// For input data structure
struct DataIn {
    char a;
    int b;
    short c;
} *dataIn;

// Arithmetic funciton
static void drv_arithmetic_routine(struct work_struct* ws);

// Input and output data from/to DMA
void myoutc(unsigned char data,unsigned short int port) {
    *(volatile unsigned char*)(dma_buf+port) = data;
}
void myouts(unsigned short data,unsigned short int port) {
    *(volatile unsigned short*)(dma_buf+port) = data;
}
void myouti(unsigned int data,unsigned short int port) {
    *(volatile unsigned int*)(dma_buf+port) = data;
}
unsigned char myinc(unsigned short int port) {
    return *(volatile unsigned char*)(dma_buf+port);
}
unsigned short myins(unsigned short int port) {
    return *(volatile unsigned short*)(dma_buf+port);
}
unsigned int myini(unsigned short int port) {
    return *(volatile unsigned int*)(dma_buf+port);
}

//this function is used to open the device
static int drv_open(struct inode* ii, struct file* ff) {
	try_module_get(THIS_MODULE);
    printk("%s:%s(): device open\n", PREFIX_TITLE, __func__);
	return 0;
}

//this function is used to close the device
static int drv_release(struct inode* ii, struct file* ff) {
	module_put(THIS_MODULE);
    printk("%s:%s(): device close\n", PREFIX_TITLE, __func__);
	return 0;
}

//this function is used to read the result from the device
static ssize_t drv_read(struct file *filp, char __user *buffer, size_t ss, loff_t* lo) {
	/* Implement read operation for your device */
	int ret;
	
	//get the result from the corresponding part in DMA
	ret = myini(DMAANSADDR);
	
	//clean the result from the corresponding part in DMA
	myouti(0,DMAANSADDR);
	
	//make the read flag in DMA to be unreadable
	myouti(0,DMAREADABLEADDR);
	
	//put the result from the device to the user
	put_user(ret, (int*)buffer);
	printk("%s:%s:  ans = %d\n",PREFIX_TITLE,__func__,ret);
	
	return 0;
}

//write contents to the device 
static ssize_t drv_write(struct file *filp, const char __user *buffer, size_t ss, loff_t* lo) {
	/* Implement write operation for your device */
	//put the data from the user to the device
	struct DataIn data;
	copy_from_user(&data, (struct DataIn __user *)buffer, sizeof(data));
	
	//put the data from the data structure to the DMA
	myoutc(data.a, DMAOPCODEADDR);
	myouti(data.b, DMAOPERANDBADDR);
	myouts(data.c, DMAOPERANDCADDR);
	
	//use special way to call the function to operate
	INIT_WORK(work,drv_arithmetic_routine);
	printk("%s:%s(): queue work\n",PREFIX_TITLE,__func__);
	
	//judge blocking or non-blocking
	//blocking
	if (myini(DMABLOCKADDR)==1){
		printk("%s:%s(): block\n",PREFIX_TITLE,__func__);
		schedule_work(work);
    	flush_scheduled_work();
	} 
	//non-blocking
	else{
		myouti(UNREADABLE, DMAREADABLEADDR);
		schedule_work(work);
	}	
	return 0;
}

//get command from the user and change and get the information of the device
static long drv_ioctl(struct file *filp, unsigned int cmd, unsigned long arg) {
	/* Implement ioctl setting for your device */
	//get the data from the user
	int information;
	copy_from_user(&information, (int*)arg, sizeof(int));

	//change the contents in the device
	if (cmd==HW5_IOCSETSTUID){
		myouti(information, DMASTUIDADDR);
		printk("%s: %s(): My STUID is = %d\n", PREFIX_TITLE, __func__, information);	
	} 
	else if(cmd==HW5_IOCSETRWOK){
		myouti(information, DMARWOKADDR);
		printk("%s: %s(): RW OK\n", PREFIX_TITLE, __func__);
	}
	else if(cmd==HW5_IOCSETIOCOK){
		myouti(information, DMAIOCOKADDR);
		printk("%s: %s(): IOC OK\n", PREFIX_TITLE, __func__);
	}
	else if (cmd==HW5_IOCSETIRQOK){
		myouti(information, DMAIRQOKADDR);
		printk("%s: %s(): IRC OK\n", PREFIX_TITLE, __func__);
	}
	else if (cmd==HW5_IOCSETBLOCK){
		myouti(information, DMABLOCKADDR);
		if(information == 1){
    		printk("%s: %s(): Blocking IO\n", PREFIX_TITLE, __func__);
  		}
		else{
    		printk("%s: %s(): Non-Blocking IO\n", PREFIX_TITLE, __func__);
  		}
	}
	else if(cmd==HW5_IOCWAITREADABLE){
		while (myini(DMAREADABLEADDR)==UNREADABLE){
    		msleep(1000);
  		}
		printk("%s: %s(): wait readable %d\n", PREFIX_TITLE, __func__,READABLE);	
  		put_user(1, (int*) arg);
	}
	else{
		printk("there is no such command");
	}
	return 0;
}

//this function is used to achieve the prime function
static int prime(int base, short nth)
{
    int fnd=0;
    int i, num, isPrime;

    num = base;
    while(fnd != nth) {
        isPrime=1;
        num++;
        for(i=2;i<=num/2;i++) {
            if(num%i == 0) {
                isPrime=0;
                break;
            }
        }

        if(isPrime) {
            fnd++;
        }
    }
    return num;
}

//this function is used to operate
static void drv_arithmetic_routine(struct work_struct* ws) {
	/* Implement arthemetic routine */
	//get the data and operator from the DMA
	char a = myinc(DMAOPCODEADDR);
	int b = myini(DMAOPERANDBADDR);
	short c = myins(DMAOPERANDCADDR);
	int result;
	if (a=='+'){
		result = b+c;
	}
	else if(a=='-'){
		result = b-c;
	}
	else if(a=='*'){
		result = b*c;
	}
	else if(a=='/'){
		result = b/c;
	}
	else if(a=='p'){
		result = prime(b,c);
	}
	else{
		result = 0;
		printk("there is no such operator");
	}

	printk("%s: %s(): %d %c %d = %d\n", PREFIX_TITLE, __func__,b,a,c,result);
	
	//store the result into the DMA
	myouti(result, DMAANSADDR);
	
	//set the read flag in the DMA to be readable
	myouti(READABLE, DMAREADABLEADDR);
}

static irqreturn_t handler(int irq, void* dev_id){
	interrupt_num ++;
  	return IRQ_HANDLED;
}

//this function is used to init the modules
static int __init init_modules(void) {

	dev_t dev;
	    
	printk("%s:%s():...............Start...............\n", PREFIX_TITLE, __func__);
	//init the count of the interrupt to be 0
	interrupt_num = 0;
	request_irq(IRQ_NUM, handler, IRQF_SHARED, "interrupt", (void*)dev_id);
	printk("%s:%s(): request_irq %d return %d\n", PREFIX_TITLE, __func__,IRQ_NUM,interrupt_num);
	
	dev_cdev = cdev_alloc();
	
	/* Register chrdev */ 
	if(alloc_chrdev_region(&dev, DEV_BASEMINOR, DEV_COUNT, DEV_NAME) < 0) {
		printk(KERN_ALERT"register chrdev failed!\n");
		return -1;
    } else {
		printk("%s:%s(): register chrdev(%i,%i)\n", PREFIX_TITLE, __func__, MAJOR(dev), MINOR(dev));
    }

    dev_major = MAJOR(dev);
    dev_minor = MINOR(dev);
    
	/* Init cdev and make it alive */
	dev_cdev->ops = &fops;
    dev_cdev->owner = THIS_MODULE;
    if(cdev_add(dev_cdev, dev, 1) < 0) {
		printk(KERN_ALERT"Add cdev failed!\n");
		return -1;
   	}
    
	/* Allocate DMA buffer */
	dma_buf = kzalloc(DMA_BUFSIZE, GFP_KERNEL);
	printk("%s: %s(): allocate dma buffer\n",PREFIX_TITLE, __FUNCTION__);

	/* Allocate work routine */
	work = kmalloc(sizeof(typeof(*work)), GFP_KERNEL);
	
	return 0;
}

//this function is used to exit the modules
static void __exit exit_modules(void) {
	printk("%s: %s(): interrupt count=%d\n", PREFIX_TITLE, __FUNCTION__, interrupt_num);
	//free the irq
	free_irq(IRQ_NUM, (void *)dev_id);
	
	/* Free DMA buffer when exit modules */
	kfree(dma_buf);
	printk("%s: %s(): free dma buffer\n",PREFIX_TITLE, __FUNCTION__);
	
	/* Delete character device */
	unregister_chrdev_region(MKDEV(dev_major,dev_minor), DEV_COUNT);
	cdev_del(dev_cdev);

	/* Free work routine */
	kfree(work);
	
	printk("%s:%s(): unregister chrdev\n", PREFIX_TITLE, __func__);
	printk("%s:%s():..............End..............\n", PREFIX_TITLE, __func__);
}

module_init(init_modules);
module_exit(exit_modules);
