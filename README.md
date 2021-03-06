Median Filter
=============

### Author

- Petar Kaselj
- Marijan Simundic Bendic
- Eugen Vrlic

### Description


Advanced Computer Architecture Project

Image median filter implemented on CPU, GPU (CUDA, CUDA w/ shared memory)

---

### Dependencies

-	[OpenCV 4.5.5](https://opencv.org/releases/)
	- opencv_world455 (.dll | .pdb )
	- opencv_world455d (.dll | .pbd )
	
-	MS Visual Studio Community 2019 (Version 16.11.10)
-	CUDA 11.5 or newer
---

### Preparation


1. Clone repository

	```
		git clone git@github.com:pkaselj/MedianFilter.git
	```
    
2. Download and install [OpenCV 4.5.5](https://opencv.org/releases/)

3. Copy
	```
        <OpenCV folder>\build\x64\vc15\bin\opencv_world455.dll ---> <Project Folder>\external\dll\release
        <OpenCV folder>\build\x64\vc15\bin\opencv_world455.pdb ---> <Project Folder>\external\dll\release
        <OpenCV folder>\build\x64\vc15\bin\opencv_world455d.dll ---> <Project Folder>\external\dll\debug
        <OpenCV folder>\build\x64\vc15\bin\opencv_world455d.dll ---> <Project Folder>\external\dll\debug
    ```

4. Add CUDA as a build dependecy to project:
	- In Visual Studio 2019 Solution Explorer:
	- Right click on _MedianFilter_ project
	- _Build dependencies_ > _Build customizations..._
	- Select _CUDA 11.5(.targets, .props)_ (or newer)
	- Confirm by clicking _OK_

---

### Execution

In Visual Studio select profile ```Release x64``` or profile ```Debug x64``` and click ```Start``` or ```Local Windows Debugger```
