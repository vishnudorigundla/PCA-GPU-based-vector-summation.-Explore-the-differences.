# PCA-GPU-based-vector-summation.-Explore-the-differences.
i) Using the program sumArraysOnGPU-timer.cu, set the block.x = 1023. Recompile and run it. Compare the result with the execution confi guration of block.x = 1024. Try to explain the difference and the reason.

ii) Refer to sumArraysOnGPU-timer.cu, and let block.x = 256. Make a new kernel to let each thread handle two elements. Compare the results with other execution confi gurations.
## Aim:

## Procedure:

## Output:

## Result:
1024


![Screenshot from 2023-04-28 09-13-30](https://user-images.githubusercontent.com/94175324/235049538-0c7fef68-dcf5-4109-9492-f34cd6275e00.png)


1023


![Screenshot from 2023-04-28 09-25-21](https://user-images.githubusercontent.com/94175324/235050576-1a2d188a-6c1b-4d9d-8f7e-481000283d52.png)

256

![Screenshot from 2023-04-28 09-29-18](https://user-images.githubusercontent.com/94175324/235051102-d14cf857-5031-43b0-a7ed-98cf1fe7f4d6.png)
