#include "virtual_memory.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__device__ void init_invert_page_table(VirtualMemory *vm) {
  for (u32 i = 0; i < vm->PAGE_ENTRIES; i++) {
    vm->invert_page_table[i] = 0x80000000; // invalid := MSB is 1
    vm->invert_page_table[i + vm->PAGE_ENTRIES] = i;
 }
}

__device__ void vm_init(VirtualMemory *vm, uchar *buffer, uchar *storage,
                        u32 *invert_page_table, int *pagefault_num_ptr,
                        int PAGESIZE, int INVERT_PAGE_TABLE_SIZE,
                        int PHYSICAL_MEM_SIZE, int STORAGE_SIZE,
                        int PAGE_ENTRIES) {
  // init variables
  vm->buffer = buffer;
  vm->storage = storage;
  vm->invert_page_table = invert_page_table;
  vm->pagefault_num_ptr = pagefault_num_ptr;

  // init constants
  vm->PAGESIZE = PAGESIZE;
  vm->INVERT_PAGE_TABLE_SIZE = INVERT_PAGE_TABLE_SIZE;
  vm->PHYSICAL_MEM_SIZE = PHYSICAL_MEM_SIZE;
  vm->STORAGE_SIZE = STORAGE_SIZE;
  vm->PAGE_ENTRIES = PAGE_ENTRIES;

  // before first vm_write or vm_read
  init_invert_page_table(vm);
}

__device__ uchar vm_read(VirtualMemory *vm, u32 addr) {
  /* Complate vm_read function to read single element from data buffer */
	u32 page_number = addr / vm->PAGESIZE;
	u32 page_offset = addr % vm->PAGESIZE;
	u32 frame_number;
	bool condition = false;
	for (u32 i = 0; i < vm->PAGE_ENTRIES; i++) {
		if (vm->invert_page_table[i + vm->PAGE_ENTRIES] == page_number) {
			condition = true;
			vm->invert_page_table[i] = 0;
			frame_number = i;
			break;
		}
	}
	if (condition == false) {
		frame_number = LRU_swap(vm, page_number);
	}
	for (u32 i = 0; i < vm->PAGE_ENTRIES; i++) {
		if (vm->invert_page_table[i] != 0x80000000)
		{
			vm->invert_page_table[i] += 1;
		}
	}
	return vm->buffer[frame_number*vm->PAGESIZE+page_offset]; //TODO
}

__device__ void vm_write(VirtualMemory *vm, u32 addr, uchar value) {
  /* Complete vm_write function to write value into data buffer */
	u32 page_number = addr / vm->PAGESIZE;
	u32 page_offset = addr % vm->PAGESIZE;
	u32 frame_number;
	bool condition = false;
	for (u32 i = 0; i < vm->PAGE_ENTRIES; i++) {
		if (vm->invert_page_table[i] == 0x80000000)
		{
			condition = true;
			*(vm->pagefault_num_ptr) += 1;
			vm->invert_page_table[i] = 0;
			vm->invert_page_table[i + vm->PAGE_ENTRIES] == page_number;
			frame_number = i;
			break;
		}
		else if ((vm->invert_page_table[i] != 0x80000000)&&(vm->invert_page_table[i + vm->PAGE_ENTRIES] == page_number))
		{
			condition = true;
			vm->invert_page_table[i] = 0;
			frame_number = i;
			break;
		}
	}
	if (condition == false) {
		frame_number = LRU_swap(vm,page_number);
	}
	for (u32 i = 0; i < vm->PAGE_ENTRIES; i++) {
		if (vm->invert_page_table[i] != 0x80000000)
		{
			vm->invert_page_table[i] += 1;
		}
	}
	vm->buffer[frame_number*vm->PAGESIZE+page_offset] = value;

}

__device__ void vm_snapshot(VirtualMemory *vm, uchar *results, int offset,
                            int input_size) {
  /* Complete snapshot function togther with vm_read to load elements from data
   * to result buffer */
	for (u32 i = offset; i < input_size; i++) {
		uchar result = vm_read(vm, i);
		*(results + i) = result;
	}
}


__device__ u32 LRU_swap(VirtualMemory *vm,u32 pageNum) {
	*(vm->pagefault_num_ptr) += 1;
	u32 index = 0;
	for (u32 i = 0; i < vm->PAGE_ENTRIES; i++) {
		if (vm->invert_page_table[i] > vm->invert_page_table[index]) {
			index = i;
		}
	}
	for (u32 i = 0; i < vm->PAGESIZE; i++) {
		vm->storage[vm->invert_page_table[index + vm->PAGE_ENTRIES] * vm->PAGESIZE + i] = vm->buffer[index*vm->PAGESIZE + i];
		vm->buffer[index*vm->PAGESIZE + i] = vm->storage[pageNum*vm->PAGESIZE + i];
	}
	vm->invert_page_table[index + vm->PAGE_ENTRIES] = pageNum;
	u32 frameNum = index;
	vm->invert_page_table[index] = 0;
	return frameNum;
}
