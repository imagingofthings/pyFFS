---
title: 'pyFFS: A Python Library for Fast Fourier Series Computation and Interpolation with GPU Acceleration'
tags:
  - Python
  - fast Fourier series
  - bandlimited interpolation
  - chirp Z-transform
  - GPU
authors:
  - name: Eric Bezzam
    orcid: 0000-0003-4837-5031
    affiliation: 1
  - name: Sepand Kashani
    orcid: 0000-0002-0735-371X
    affiliation: 1
  - name: Paul Hurley
    affiliation: 2
  - name: Martin Vetterli
    orcid: 0000-0002-6122-1216
    affiliation: 1
  - name: Matthieu Simeoni
    orcid: 0000-0002-4927-3697
    affiliation: 3
affiliations:
 - name: Audiovisual Communications Laboratory, École Polytechnique Fédérale de Lausanne
   index: 1
 - name: International Centre for Neuromorphic Systems, Western Sydney University
   index: 2
 - name: Centre for Imaging, École Polytechnique Fédérale de Lausanne
   index: 3
date: 11 November 2021
bibliography: paper.bib
   
---

# Summary

Fourier transforms are an often necessary component in many computational tasks,
and can be computed efficiently through the fast Fourier transform (FFT) 
algorithm [@Cooley1965]. However, many applications involve an underlying 
continuous signal, and a more natural choice would be to work with e.g. the 
Fourier series (FS) coefficients in order to avoid the additional overhead of 
translating between the analog and discrete domains. Unfortunately, there exists
very little literature and tools for the manipulation of FS coefficients from 
discrete samples in practice. This paper introduces a Python library called 
pyFFS for efficient FS coefficient computation, convolution, and interpolation 
for N-D dimensional signals. 
While the libraries SciPy [@Virtanen2020] and NumPy [@Harris2020] provide 
efficient functionality for discrete Fourier transform coefficients via the FFT 
algorithm, pyFFS addresses the computation of FS coefficients through what we 
call the fast Fourier series (FFS). Moreover, pyFFS includes an FS interpolation
method based on the chirp Z-transform (CZT) [@rabiner1969chirp] that can make it
more than an order of magnitude faster than the SciPy equivalent when one wishes
to perform distortionless bandlimited interpolation. GPU support through the 
CuPy library [@Okuta2017] allows for further acceleration, e.g. an order of 
magnitude faster for computing the 2-D FS coefficients of $1000\times1000$ 
samples and nearly two orders of magnitude faster for 2-D interpolation.


# Statement of need

So why use pyFFS rather than the FFT routines from NumPy/SciPy? One reason is 
convenience when working with continuous-domain signals. The philosophy of pyFFS
is to retain the continuous-domain perspective, often neglected when using 
numerical libraries such as NumPy and SciPy, which allows for much clearer code
(as will be shown in a Fourier optics example). This can prevent common pitfalls
due to an invalid conversion between discrete and continuous domains. Another
reason is efficiency. We benchmark pyFFS with equivalent functions in SciPy, 
observing scenarios in which the proposed library is more than an order of 
magnitude faster, i.e. for 2-D convolution and for interpolation. Moreover, GPU
support has been seamlessly incorporated for an even faster implementation. Just
as the FFT implementation via NumPy and SciPy can be readily used for an 
efficient $\mathcal{O}(N\log N)$ analysis and synthesis of discrete sequences, 
pyFFS offers the same ease-of-use and performance capabilities for discrete
representations of continuous-domain signals, along with faster interpolation
techniques.

pyFFS has been used in [@fageot2020tv], where the authors make use of the
FFS algorithm to efficiently compute multidimensional periodic splines. It is 
currently being used in radio-astronomy and optics projects, in which physical 
processes have been modeled as bandlimited. In general, pyFFS is useful for 
interpolating multidimensional functions efficiently. Such functions are 
commonplace in numerous applications, e.g. radio-astronomy, medical imaging, 
holography, etc.

# <a name="comparison"></a>Comparison with SciPy

Below is a functionality comparison between pyFFS and SciPy [@Virtanen2020].

|   | pyFFS | SciPy |
|:-:|:-:|:-:|
| 1-D Fourier analysis | `pyffs.ffs`  |  `scipy.fft.fft` |
| 1-D Fourier synthesis | `pyffs.iffs`  | `scipy.fft.ifft`  |
| N-D Fourier analysis  | `pyffs.ffsn`  |  `scipy.fft.fftn` |
| N-D Fourier synthesis |  `pyffs.iffsn` |  `scipy.fft.ifftn` |
| N-D convolution |  `pyffs.convolve` |  `scipy.signal.fftconvolve` |
| 1-D bandlimited interpolation |  `pyffs.fs_interp` |  `scipy.signal.resample` |
| N-D bandlimited interpolation |  `pyffs.fs_interpn` | - |

Note that SciPy's `scipy.fft.fftconvolve` zero-pads inputs in order to
approximate a linear convolution, while pyFFS performs a circular convolution.
Within SciPy, circular convolution is only supported for 2-D by calling 
`scipy.signal.convolve2d`  with the parameter `boundary = wrap`. This method can
be considerably slower than pyFFS' `pyffs.convolve` for modest size inputs, as
shown below. All benchmarking is performed on a Lenovo ThinkPad P15 Gen 1 
laptop, with an Intel i7-10850H six-core processor.

![2-D bandlimited circular convolution.\label{fig:profile_convolve2d}](fig/profile_convolve2d.png){width=50%}

For N-D bandlimited interpolation with SciPy, it is possible to use 
`scipy.signal.resample` along each dimension. However, there is no one-shot
function. Below we benchmark 1-D and 2-D interpolation as we vary the 
width of the interval over which we interpolate.

![Bandlimited interpolation: (left) 1-D and (right) 2-D.\label{fig:bandlimited_interpolation}](fig/bandlimited_interpolation.png){width=100%}

As can be observed above, the complexity of pyFFS's interpolation stays constant
as we vary the size of the interpolation region. That is because the CZT-based 
approach of pyFFS performs interpolation solely in the region of interest. 
SciPy's interpolation, on the other hand, interpolates by zero-padding the DFT
coefficients and taking an inverse DFT over the full period, requiring 
operations outside the region of interest. The complexity of SciPy's approach is
$\mathcal{O}(N_t\log N_t)$, where $N_t=\lceil T/\Delta t\rceil$, $T$ is the 
period, and $\Delta t$ is the desired resolution. The complexity of the pyFFS 
method is defined by that of the CZT [@rabiner1969chirp], namely 
$\mathcal{O}((M+N)\log (M+N))$ where $M$ is the number of interpolation points 
and $N$ is the number of FS coefficients. For a mathematical presentation and
proof of the proposed CZT-based interpolation, we refer to 
[this technical report](https://arxiv.org/abs/2110.00262).

In the benchmark below, we interpolate a $2\%$ region of a 2-D function, of 
which we have $256\times 256$ samples. As we vary the resolution of the
interpolation, we notice that the pyFFS function is consistently more efficient.

![2-D bandlimited interpolation: (left) profiling increasing resolution for 2% region; (middle) visualization of corresponding region; (right) 2% region of middle plot.\label{fig:bandlimited_interp1d_vary_width}](fig/interp_increasing_resolution.png){width=100%}



# GPU acceleration

CuPy [@Okuta2017] supports many of NumPy and SciPy's functionality in order to 
perform equivalent operations on CUDA. pyFFS' functions can be used for both CPU 
and CUDA by simply passing NumPy or CuPy arrays (respectively) as inputs to the 
corresponding pyFFS functions. Below is a functionality comparison between pyFFS
and CuPy.

|   | pyFFS | CuPy |
|:-:|:-:|:-:|
| 1-D Fourier analysis | `pyffs.ffs`  |  `cupyx.scipy.fft.fft` |
| 1-D Fourier synthesis | `pyffs.iffs`  | `cupyx.scipy.fft.ifft`  |
| 1-D Fourier synthesis | `pyffs.iffs`  | `cupyx.scipy.fft.ifft`  |
| N-D Fourier analysis  | `pyffs.ffsn`  |  `cupyx.scipy.fft.fftn` |
| N-D Fourier synthesis |  `pyffs.iffsn` |  `cupyx.scipy.fft.ifftn` |
| N-D convolution |  `pyffs.convolve` |  `cupyx.scipy.signal.fftconvolve` |
| 1-D bandlimited interpolation |  `pyffs.fs_interp` | - |
| N-D bandlimited interpolation |  `pyffs.fs_interpn` | - |

At the time of writing, CuPy has not implemented an equivalent of SciPy's 
`resample` function. To the best of our knowledge, pyFFS is the only Python 
library offering GPU support for bandlimited interpolation.

There are two important considerations when using a GPU. Firstly, if the 
application permits, it is recommended to work with `float32`/ `complex64`
arrays for less memory consumption and potentially faster computation. By 
default, NumPy and CuPy create `float64` / `complex128` arrays, e.g. when 
initializing an array with `np.zeros`, so casting the arrays accordingly is 
recommended. In the benchmarking tests below, we use `float32`/ `complex64`
arrays. Secondly, the benefits of using a GPU typically emerge when the
processed arrays are larger than the CPU cache. So the crossover between CPU and
GPU performance can be very hardware-dependent. All benchmarking below is 
performed on a Lenovo ThinkPad P15 Gen 1 laptop, with an Intel i7-10850H 
six-core processor and an NVIDIA Quadro RTX 3000 GPU.

Figure \Ref{fig:gpu_ffs} compares the processing time between a CPU and a GPU for 
computing an increasing number of FS coefficients.

![Profiling Fourier series computation: (left) 1-D and (right) 2-D.\label{fig:gpu_ffs}](fig/gpu_ffs.png){width=100%}

In 1-D, for more than $1'000$ coefficients it starts to become beneficial to use
a GPU, and at around $10'000$ coefficients it is an order of magnitude faster to 
use a GPU. In 2-D, the crossover point is at around $100$ coefficients per 
dimension, and at around $1'000$ coefficients per dimension it is more than an 
order of magnitude faster to use a GPU. From the 1-D and 2-D cases, it is clear 
that using a GPU scales well as the input increases in size. When considering a 
2-D or even a 3-D object, where input sizes quickly grow, it is attractive to 
make use of a GPU for even modest input sizes.

Figure \ref{fig:profile_interp1d_gpu} profiles the processing time for 1-D FS
interpolation. Using a GPU becomes more attractive as the number of coefficients
and number of samples exceeds $300$. As mentioned earlier, this is probably the
point when the arrays and computation can no longer fit on the CPU cache.

![Profiling 1-D Fourier series interpolation.\label{fig:profile_interp1d_gpu}](fig/profile_interp1d_gpu.png){width=100%}

Figure \ref{fig:profile_interp2d_gpu} profiles the processing time for 2-D FS
interpolation. Using a GPU consistently provides two orders of magnitude faster
computation for a varying number of FS coefficients and varying number of
interpolation points per dimension. The benefits of using a GPU are even more
prominent in 2-D as input sizes quickly grow when considering multidimensional
scenarios.

![Profiling 2-D Fourier series interpolation.\label{fig:profile_interp2d_gpu}](fig/profile_interp2d_gpu.png){width=100%}


# <a name="optics"></a>Example usage (Fourier optics)

In Fourier optics, we are often interested in the propagation of light between
two planes, i.e. a source plane and a target plane as shown in Figure
\ref{fig:fourier_optics}(a). Given an aperture function or phase pattern at the
source plane, we would like to determine the pattern at the target plane, as
predicted by the Rayleigh-Sommerfeld diffraction formula. This propagation is 
often modeled with one of three approaches that make use of the FFT for an 
efficient simulation: Fraunhofer approximation, Fresnel approximation, or the 
angular spectrum method [@Goodman2005]. The choice between these three 
approaches typically depends on the requirements of the application, e.g. the 
distance between the two planes, the desired sampling rate at the input or 
output, and the size of input and output regions [@Schmidt2010]. For all 
approaches, we again find ourselves with a continuous-domain phenomenon that can
be considered bandlimited and periodic. Bandlimited as in practice we consider 
finite input and output regions, lending to a restricted set of angles and 
therefore a bandlimited spatial frequency response between the source and target
planes. This restriction of angles is shown in Figure 
\ref{fig:fourier_optics}(b). Even though our input may not be bandlimited, the 
resulting output will be bandlimited after convolution with such a response 
[@Matsushima2009]. Finally, we can frame the optical simulation as periodic as 
the input and output regions have a compact support and can thus be replicated 
to form periodic signals.

![Visualization of optical wave propagation setup.\label{fig:fourier_optics}](fig/fourier_optics.png){width=100%}


The application of the CZT, or equivalently the fractional FT [@Bailey1991], for
interpolation has already found its use in Fourier optics to resample the output
plane outside of the grid defined by the FFT [@Muffoletto2007], as demonstrated 
with pyFFS in Figure \ref{fig:bandlimited_interpolation}. 

Below we show how the pyFFS interface can be used in optical wave propagation
for efficient simulation and interpolation.

    import pyffs    

    # pad input and reorder
    f_pad = numpy.pad(f, pad_width=pad_width)
    f_pad_reorder = pyffs.ffs_shift(f_pad)
    
    # compute FS coefficients of input
    F = pyffs.ffsn(f_pad_reorder, T, T_c, N_FS)
    
    # convolution in frequency domain with free space transfer function
    G = F * H
    
    # interpolate at the desired location and resolution
    # a and b specify the region while N_out specifies the resolution
    g = pyffs.fs_interpn(G, T, a, b, N_out)

The free space propagation transfer function `H` in the above code listing can
be obtained by evaluating the analytic expression for the Fresnel approximation
or the angular spectrum method transfer functions at the appropriate frequency
values [@Goodman2005].

One may wish to simulate an output window with the same size as the input but at
a finer resolution. In order to circumvent the much larger FFT that this may
require, an approach known as rectangular tiling [@Muffoletto2007], as shown in
Figure \ref{fig:fourier_optics}(c), can be used to split up the output window 
into tiles. In its original proposition, the tiles are simulated sequentially, 
but with a GPU they could be computed in parallel for a significantly shorter
simulation time: pyFFS's GPU support enables this possibility. Moreover, 
rectangular tiling in its original proposition requires that each tile has the
same number of samples as the input window. This restriction is removed by the
interpolation approach of pyFFS.


# Acknowledgements

TODO : Funding for Sepand

# References