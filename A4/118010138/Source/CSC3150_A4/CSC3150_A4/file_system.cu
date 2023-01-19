#include "file_system.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

__device__ __managed__ u32 gtime = 0;

//this part is used to calculate the create time of every file
//and it is the relative time and only used to compare
__device__ u32 created_time = 0;

//this part is used to calculate the modified time of every file
//and it is the relative time and only used to compare
__device__ u32 modified_time = 0;

__device__ u32 find_fp_using_name(FileSystem *fs, char *s) {
	for (u32 i = 0; i < fs->FCB_ENTRIES; i++) {
		//this is the initial address of every fcb block
		u32 fcb_block_start = fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE;
		//now we want to get the file name of every fcb block
		char file_name[20];
		for (u32 j = 0; j < 20; j++) {
			char part = fs->volume[fcb_block_start + j];
			file_name[j] = part;
		}
		u32 length = 0;
		while (true) {
			if (s[length] != '\0') {
				length += 1;
			}
			else {
				break;
			}
		}
		bool condition = true;

		//now we will compare the file name of every block with the input file name byte by byte
		//if any byte of them is different, we will change the condition to false
		for (u32 j = 0; j < length; j++) {
			if (file_name[j] != s[j]) {
				condition = false;
			}
		}
		if (condition == false) {
			continue;
		}
		else if (condition == true)
		{
			return fcb_block_start;
		}
	}
	//if there is no file name matched, we will return -1
	return -1;
}

//we will use a number to implement the super block and the number here is just the value which the super block number will change to
__device__ void update_super_block(FileSystem *fs, u32 number) {
	int number_RSB_1 = number - ((number >> 8) << 8);
	int number_RSB_2 = ((number - ((number >> 16) << 16) - number_RSB_1) >> 8);
	int number_RSB_3 = ((number - ((number >> 24) << 24) - number_RSB_1 - (number_RSB_2 << 8)) >> 16);
	int number_RSB_4 = number >> 24;
	fs->volume[0] = number_RSB_1;
	fs->volume[1] = number_RSB_2;
	fs->volume[2] = number_RSB_3;
	fs->volume[3] = number_RSB_4;
}

//the whole thought is to use "20+4+4+2+2" way to store the information of every file
//and 20 bytes of them have already created and we do not need to operate them
__device__ u32 update_fcb(FileSystem *fs, u32 fcb_block_start, u32 file_address, u32 size, u32 case_number) {
	//storage the file address
	int address_RSB_1 = file_address - ((file_address >> 8) << 8);
	int address_RSB_2 = ((file_address - ((file_address >> 16) << 16) - address_RSB_1) >> 8);
	int address_RSB_3 = ((file_address - ((file_address >> 24) << 24) - address_RSB_1 - (address_RSB_2 << 8)) >> 16);
	int address_RSB_4 = file_address >> 24;

	fs->volume[fcb_block_start + 20 + 0] = address_RSB_1;
	fs->volume[fcb_block_start + 20 + 1] = address_RSB_2;
	fs->volume[fcb_block_start + 20 + 2] = address_RSB_3;
	fs->volume[fcb_block_start + 20 + 3] = address_RSB_4;

	//storage the file size
	int size_RSB_1 = size - ((size >> 8) << 8);
	int size_RSB_2 = ((size - ((size >> 16) << 16) - size_RSB_1) >> 8);
	int size_RSB_3 = ((size - ((size >> 24) << 24) - size_RSB_1 - (size_RSB_2 << 8)) >> 16);
	int size_RSB_4 = size >> 24;

	fs->volume[fcb_block_start + 24 + 0] = size_RSB_1;
	fs->volume[fcb_block_start + 24 + 1] = size_RSB_2;
	fs->volume[fcb_block_start + 24 + 2] = size_RSB_3;
	fs->volume[fcb_block_start + 24 + 3] = size_RSB_4;

	//storage the file create time when the case number is 1
	if (case_number == 1) {
		int create_time_RSB_1 = created_time - ((created_time >> 8) << 8);
		int create_time_RSB_2= ((created_time - ((created_time >> 16) << 16) - create_time_RSB_1) >> 8);
		fs->volume[fcb_block_start + 28 + 0] = create_time_RSB_1;
		fs->volume[fcb_block_start + 28 + 1] = create_time_RSB_2;
		//because some other operations will follow the open operation
		//we do not need to update the modified time here
		//what we need to do is to change the modified time from NULL to 0
		fs->volume[fcb_block_start + 30 + 0] = 0;
		fs->volume[fcb_block_start + 30 + 1] = 0;
	}
	
	//storage the modified time when the case number is 2
	if (case_number == 2) {
		int modified_time_RSB_1= modified_time - ((modified_time >> 8) << 8);
		int modified_time_RSB_2 = ((modified_time - ((modified_time >> 16) << 16) - modified_time_RSB_1) >> 8);
		fs->volume[fcb_block_start + 30 + 0] = modified_time_RSB_1;
		fs->volume[fcb_block_start + 30 + 1] = modified_time_RSB_2;
	}
}


__device__ u32 get_address_from_FCB(FileSystem *fs, u32 fp) {
	u32 address = 0;
	for (u32 i = 0; i < 4; i++) {
		address += (fs->volume[fp + 20 + i] << (i * 8));
	}
	return address;
}

__device__ u32 get_size_from_FCB(FileSystem *fs, u32 fp) {
	u32 size = 0;
	for (u32 i = 0; i < 4; i++) {
		size += (fs->volume[fp + 24 + i] << (i * 8));
	}
	return size;
}

__device__ u32 get_create_time_from_FCB(FileSystem *fs, u32 fp) {
	u32 time=0;
	for (u32 i = 0; i < 2; i++) {
		time += (fs->volume[fp + 28 + i] << (i * 8));
	}
	return time;
}

__device__ u32 get_modified_time_from_FCB(FileSystem *fs, u32 fp) {
	u32 time=0;
	for (u32 i = 0; i < 2; i++) {
		time += (fs->volume[fp + 30 + i] << (i * 8));
	}
	return time;
}

__device__ u32 get_block_used_number(FileSystem *fs) {
	u32 number;
	if (fs->volume[0] == NULL) {
		return 0;
	}
	for (u32 i = 0; i < 4; i++) {
		number += (fs->volume[i] << (i * 8));
	}
	return number;
}

__device__ void fs_init(FileSystem *fs, uchar *volume, int SUPERBLOCK_SIZE,
							int FCB_SIZE, int FCB_ENTRIES, int VOLUME_SIZE,
							int STORAGE_BLOCK_SIZE, int MAX_FILENAME_SIZE, 
							int MAX_FILE_NUM, int MAX_FILE_SIZE, int FILE_BASE_ADDRESS)
{
  // init variables
  fs->volume = volume;

  // init constants
  fs->SUPERBLOCK_SIZE = SUPERBLOCK_SIZE;
  fs->FCB_SIZE = FCB_SIZE;
  fs->FCB_ENTRIES = FCB_ENTRIES;
  fs->STORAGE_SIZE = VOLUME_SIZE;
  fs->STORAGE_BLOCK_SIZE = STORAGE_BLOCK_SIZE;
  fs->MAX_FILENAME_SIZE = MAX_FILENAME_SIZE;
  fs->MAX_FILE_NUM = MAX_FILE_NUM;
  fs->MAX_FILE_SIZE = MAX_FILE_SIZE;
  fs->FILE_BASE_ADDRESS = FILE_BASE_ADDRESS;

}

//this part is used to open the file
__device__ u32 fs_open(FileSystem *fs, char *s, int op)
{
	/* Implement open operation here */
	if (op == G_READ) {
		u32 fp = find_fp_using_name(fs, s);
		if (fp != -1) {
			return fp;
		}
		else {
			return -1;
		}
	}
	else if (op == G_WRITE)
	{
		u32 fp = find_fp_using_name(fs, s);
		if (fp != -1) {
			return fp;
		}
		else {
			//this is the process of creating the fcb information for the new file
			u32 fcb_block_start = 0;
			for (u32 i = 0; i < fs->FCB_ENTRIES; i++) {
				fcb_block_start = fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE;
				if (fs->volume[fcb_block_start] != NULL) {
					continue;
				}
				else {
					created_time += 1;
					u32 used_block_number = get_block_used_number(fs);
					u32 new_addr = fs->FILE_BASE_ADDRESS + used_block_number * 32;
					u32 length = 0;
					while (true) {
						if (s[length] != '\0') {
							length += 1;
						}
						else {
							break;
						}
					}
					for (u32 i = 0; i < length; i++) {
						fs->volume[fcb_block_start + i] = s[i];
					}
					update_fcb(fs, fcb_block_start, new_addr, 0, 1);
					break;
				}
			}
			return fcb_block_start;
		}
	}
}

__device__ void fs_read(FileSystem *fs, uchar *output, u32 size, u32 fp)
{
	/* Implement read operation here */
	if (fp == -1) {
		printf("there is no such file so you can not read\n");
	}
	else {
		u32 file_address = get_address_from_FCB(fs, fp);
		for (u32 i = 0; i < size; i++) {
			output[i] = fs->volume[file_address + i];
		}
	}
}

//there are two cases of write operation
//one is that the file does not exist
//the other is that the file has already existed
__device__ u32 fs_write(FileSystem *fs, uchar* input, u32 size, u32 fp)
{
	/* Implement write operation here */
	u32 file_address = get_address_from_FCB(fs, fp);
	modified_time += 1;
	//if the file does not exist
	if (fs->volume[file_address] == NULL) {
		//update the super block
		u32 increase_block_number = (size / 32 + 1);
		u32 used_block_number = get_block_used_number(fs);
		used_block_number += increase_block_number;
		update_super_block(fs, used_block_number);
		//update the storage
		for (u32 i = 0; i < size; i++) {
			fs->volume[file_address + i] = input[i];
		}
		//update the fcb
		update_fcb(fs, fp, file_address, size, 2);
	}
	//if the file has already existed
	else {
		//delete the old contents
		u32 old_file_address = get_address_from_FCB(fs, fp);
		u32 old_file_size = get_size_from_FCB(fs, fp);
		//update the super block
		u32 decrease_block_number = (old_file_size / 32 + 1);
		u32 used_block_number = get_block_used_number(fs);
		used_block_number -= decrease_block_number;
		update_super_block(fs, used_block_number);
		//compact to update the storage
		u32 old_file_address_2 = old_file_address + decrease_block_number * 32;
		u32 compact_time = 1085440 - old_file_address_2;
		for (u32 i = 0; i < compact_time; i++) {
			fs->volume[old_file_address + i] = fs->volume[old_file_address_2 + i];
			fs->volume[old_file_address_2 + i] = NULL;
		}

		//write new contents into the file system
		//update the storage
		//get the new file address
		u32 new_addr = fs->FILE_BASE_ADDRESS + used_block_number * 32;
		for (u32 i = 0; i < size; i++) {
			fs->volume[new_addr + i] = input[i];
		}
		//update the super block
		u32 increase_block_number = (size / 32 + 1);
		used_block_number += increase_block_number;
		update_super_block(fs, used_block_number);
		//update the fcb
		update_fcb(fs, fp, new_addr, size, 2);
	}
	return 0;
}
__device__ void fs_gsys(FileSystem *fs, int op)
{
	/* Implement LS_D and LS_S operation here */
	//list all the file names in the file system and sort them according to the modified time
	if (op == LS_D) {
		printf("===sort by modified time===\n");
		u32 file_name_fp_list[1024];
		u32 index = 0;
		for (u32 i = 0; i < 1024; i++) {
			u32 fcb_block_start = fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE;
			if (fs->volume[fcb_block_start] != NULL) {
				file_name_fp_list[index] = fcb_block_start;
				index += 1;
			}
		}
		u32 file_modified_time_list[1024];
		u32 file_modified_time = 0;
		for (u32 i = 0; i < index; i++) {
			file_modified_time = get_modified_time_from_FCB(fs, file_name_fp_list[i]);
			file_modified_time_list[i] = file_modified_time;
		}
		//use bubble sort to sort
		for (u32 i = 0; i < index - 1; i++) {
			for (u32 j = 0; j < index -1 - i; j++) {
				if (file_modified_time_list[j] < file_modified_time_list[j + 1]) {
					u32 temp_1 = file_modified_time_list[j];
					file_modified_time_list[j] = file_modified_time_list[j + 1];
					file_modified_time_list[j + 1] = temp_1;
					u32 temp_2 = file_name_fp_list[j];
					file_name_fp_list[j] = file_name_fp_list[j + 1];
					file_name_fp_list[j + 1] = temp_2;
				}
			}
		}
		//print the file names in order
		for (u32 i = 0; i < index; i++) {
			u32 fcb_address = file_name_fp_list[i];
			char file_name[20];
			for (u32 j = 0; j < 20; j++) {
				char part = fs->volume[fcb_address + j];
				file_name[j] = part;
			}
			printf("%s\n", file_name);
		}
	}
	//list all the file names in the file system according to the file size and create time
	else if (op == LS_S) {
		printf("===sort by file size===\n");
		u32 file_name_fp_list[1024];
		u32 index = 0;
		for (u32 i = 0; i < 1024; i++) {
			u32 fcb_block_start = fs->SUPERBLOCK_SIZE + i * fs->FCB_SIZE;
			if (fs->volume[fcb_block_start] != NULL) {
				file_name_fp_list[index] = fcb_block_start;
				index += 1;
			}
		}
		u32 file_size_list[1024];
		u32 file_size = 0;
		for (u32 i = 0; i < index; i++) {
			file_size = get_size_from_FCB(fs, file_name_fp_list[i]);
			file_size_list[i] = file_size;
		}

		u32 file_create_time_list[1024];
		u32 file_create_time = 0;
		for (u32 i = 0; i < index; i++) {
			file_create_time = get_create_time_from_FCB(fs, file_name_fp_list[i]);
			file_create_time_list[i] = file_create_time;
		}

		//use bubble sort to sort
		for (u32 i = 0; i < index - 1; i++) {
			for (u32 j = 0; j < index - 1 - i; j++) {
				if (file_size_list[j] < file_size_list[j + 1]) {
					u32 temp_1 = file_size_list[j];
					file_size_list[j] = file_size_list[j + 1];
					file_size_list[j + 1] = temp_1;
					u32 temp_2 = file_name_fp_list[j];
					file_name_fp_list[j] = file_name_fp_list[j + 1];
					file_name_fp_list[j + 1] = temp_2;
					u32 temp_3 = file_create_time_list[j];
					file_create_time_list[j] = file_create_time_list[j + 1];
					file_create_time_list[j + 1] = temp_3;
				}
			}
		}
		//use bubble sort to second sort
		for (u32 i = 0; i < index - 1; i++) {
			for (u32 j = 0; j < index - 1 - i; j++) {
				if ((file_create_time_list[j] > file_create_time_list[j + 1])&&(file_size_list[j] == file_size_list[j + 1])) {
					u32 temp_1 = file_size_list[j];
					file_size_list[j] = file_size_list[j + 1];
					file_size_list[j + 1] = temp_1;
					u32 temp_2 = file_name_fp_list[j];
					file_name_fp_list[j] = file_name_fp_list[j + 1];
					file_name_fp_list[j + 1] = temp_2;
					u32 temp_3 = file_create_time_list[j];
					file_create_time_list[j] = file_create_time_list[j + 1];
					file_create_time_list[j + 1] = temp_3;
				}
			}
		}
		//print the file names and sizes in order
		for (u32 i = 0; i < index; i++) {
			u32 fcb_address = file_name_fp_list[i];
			char file_name[20];
			for (u32 j = 0; j < 20; j++) {
				char part = fs->volume[fcb_address + j];
				file_name[j] = part;
			}
			printf("%s %d\n", file_name,file_size_list[i]);
		}
	}
}

__device__ void fs_gsys(FileSystem *fs, int op, char *s)
{
	/* Implement rm operation here */
	u32 fp = find_fp_using_name(fs, s);
	//if the file to delete does not exist
	if (fp == -1) {
		printf("there is no such file so you cannot remove it\n");
	}
	//if the file to delete exists
	else {
		//update the super block
		u32 file_size = get_size_from_FCB(fs, fp);
		u32 decrease_block_number = file_size / 32 + 1;
		u32 used_block_number = get_block_used_number(fs);
		used_block_number -= decrease_block_number;
		update_super_block(fs, used_block_number);
		//update the storage
		u32 file_address = get_address_from_FCB(fs, fp);
		u32 file_address_2 = file_address + decrease_block_number * 32;
		u32 compact_time = 1085440 - file_address_2;
		for (u32 i = 0; i < compact_time; i++) {
			fs->volume[file_address + i] = fs->volume[file_address_2 + i];
			fs->volume[file_address_2 + i] = NULL;
		}
		//update the fcb
		for (u32 i = 0; i < 32; i++) {
			fs->volume[fp + i] = NULL;
		}
	}
}
