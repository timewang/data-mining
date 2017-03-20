# python版本
def binary_search(arr,start,end,hkey):
	if start > end:
		return -1
	mid = int(start + (end - start)/2)
	if arr[mid] > hkey:
		return binary_search(arr, start, mid - 1, hkey)
	if arr[mid] < hkey:
		return binary_search(arr, mid + 1, end, hkey)
	return mid

test_data = [0,1,2,3,4,5,6,7,8,9,10,55,99,400,401,456,489,478,896,4569]
print(len(test_data))
print(binary_search(test_data, 0, len(test_data), 400))