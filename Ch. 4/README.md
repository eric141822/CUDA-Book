# Chapter 4. Compute architecture and scheduling

A mostly conceptial chapter that covers the basics of how CUDA works under the hood. Covering topics such as SMs (streaming multiprocessors) barrier synchronization, control divergence, warps, latency hiding (latency tolerance), zero-overhead thread scheduling... etc.

`devprops.cu` shows you how to query the device properties of your GPU. This is useful for determining the number of SMs, the number of threads per block... etc. so you can optimize your code for your specific GPU.
