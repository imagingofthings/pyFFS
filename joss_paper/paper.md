---
title: 'pyFFS: A Python Library for Fast Fourier Series Computation and Interpolation with GPU Acceleration'
tags:
  - Python
  - fast Fourier series
  - bandlimited interpolation
  - chirp Z-transform
  - GPU
authors:
  - name: Eric Bezzam^[corresponding author]
    orcid: 0000-0003-4837-5031
    affiliation: 1
  - name: Sepand Kashani
    affiliation: 1
  - name: Paul Hurley
    affiliation: 2
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
algorithm. However, many applications involve an underlying continuous signal, 
and a more natural choice would be to work with e.g. the Fourier series (FS) 
coefficients in order to avoid the additional overhead of translating between 
the analog and discrete domains. Unfortunately, there exists very little 
literature and tools for the manipulation of FS coefficients from discrete 
samples. This paper introduces a Python library called pyFFS for efficient FS 
coefficient computation, convolution, and interpolation. While the libraries 
SciPy [@Virtanen2020] and NumPy [@Harris2020] provide efficient functionality
for discrete Fourier transform coefficients via the FFT algorithm, pyFFS 
addresses the computation of FS coefficients through what we call the fast 
Fourier series (FFS). Moreover, pyFFS includes an FS interpolation method based
on the chirp Z-transform [@rabiner1969chirp] that can make it more than an order
of magnitude faster than the SciPy equivalent when one wishes to perform 
interpolation. GPU support through the CuPy library [@Okuta2017] allows for 
further acceleration, e.g. an order of magnitude faster for computing the 2-D FS
coefficients of $ 1000\times1000 $ samples and nearly two orders of magnitude 
faster for 2-D interpolation.


# Statement of need

So why use pyFFS rather than the FFT routines from NumPy/SciPy? One reason is 
convenience when working with continuous-domain signals. The philosophy of pyFFS
is to retain the continuous-domain perspective, often neglected when using 
numerical libraries such as NumPy and SciPy, which allows for much clearer code 
as seen [below](#optics). This can prevent common pitfalls due to an invalid 
conversion between discrete and continuous domains. Another reason is 
efficiency. We benchmark pyFFS with equivalent functions in SciPy, observing
scenarios in which the proposed library is more than an order of magnitude
faster, i.e. for 2-D convolution and for interpolation (results 
[below](comparison)). Moreover, GPU support has been seamlessly incorporated for
an even faster implementation. Just as the FFT implementation via NumPy and
SciPy can be readily used for an efficient $ \mathcal{O}(N \log N) $ analysis 
and synthesis of discrete sequences, pyFFS offers the same ease-of-use and
performance capabilities for discrete representations of continuous-domain
signals, with some faster interpolation tricks.

pyFFS has been used in [@fageot2020tv], where the authors make use of the
FFS algorithm to efficiently compute multidimensional periodic splines.
**TODO: discuss application in radio-astronomy and optics**. Most notable 
applications in interpolation multidimensional functions  efficiently and 
accurately, e.g. radio-astronomy, medical imaging, holography.

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
| N-D bandlimited interpolation |  `pyffs.ff_interpn` | - |

Note that SciPy's `scipy.fft.fftconvolve` zero-pads inputs in order to
approximate a linear convolution, while pyFFS performs a circular convolution.
Within SciPy, circular convolution is only supported for 2-D by calling 
`scipy.signal.convolve2d`  with the parameter `boundary-wrap`. This method can
be considerably slower than pyFFS' `pyffs.convolve` for modest size inputs, as
shown below.

![2-D bandlimited circular convolution.\label{fig:convolve2d}](fig/profile_convolve2d.png)

For N-D bandlimited interpolation with SciPy, it is possible to use 
`scipy.signal.resample` along each dimension. However, there is no one-shot
function. Below we benchmark 1-D and 2-D interpolation as we vary the 
width of the interval over which we interpolate.

![1-D bandlimited interpolation.\label{fig:bandlimited_interp1d_vary_width}](fig/bandlimited_interp1d_vary_width.png)

![2-D bandlimited interpolation.\label{fig:bandlimited_interp2d_vary_width}](fig/bandlimited_interp2d_vary_width.png)


# GPU acceleration

![1-D FS computation.\label{fig:profile_fs_interp1d_vary_NFS}](fig/profile_fs_interp1d_vary_NFS.png)

![2-D FS computation interpolation.\label{fig:profile_fs_interp1d_vary_M}](fig/profile_fs_interp1d_vary_M.png)

![1-D bandlimited interpolation, varying number of coefficients.\label{fig:profile_fs_interp1d_vary_NFS}](fig/profile_fs_interp1d_vary_NFS.png)

![1-D bandlimited interpolation, varying number of interpolation points.\label{fig:profile_fs_interp1d_vary_M}](fig/profile_fs_interp1d_vary_M.png)

![2-D bandlimited interpolation, varying number of coefficients.\label{fig:profile_fs_interp2d_vary_NFS}](fig/profile_fs_interp2d_vary_NFS.png)

![2-D bandlimited interpolation, varying number of interpolation points.\label{fig:profile_fs_interp2d_vary_M}](fig/profile_fs_interp2d_vary_M.png)


# <a name="optics"></a>Optics example

Examples when we need efficient and accurate interpolation.

Simulating optical free space propagation with pyFFS.

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


# Acknowledgements


# References