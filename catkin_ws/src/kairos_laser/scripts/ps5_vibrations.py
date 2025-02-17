#! /usr/bin/env python

#-------------------------------------------------------------------------------------------------
#   @file       : ps5_vibrations.py
#   @author     : PARAM D SALUNKHE | UTARI - AIS
#   @comments   : Proof of concept for vibration patterns using the idea of 
#                 Amplitude Modultaion and Frequency Modulation.
#-------------------------------------------------------------------------------------------------
 
#-------------------------------------------------------------------------------------------------
#   Package Imports
#-------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

#-------------------------------------------------------------------------------------------------
#   Main Function
#-------------------------------------------------------------------------------------------------

# Defining time axis t
T = 1                   # Final time
sampling_rate = 10000   # Number of samples per second
t = np.linspace(0, T, int(T * sampling_rate), endpoint=False)

# Carrier signal c(t) = Ac*cos(2*pi*f_c*t)
A_c = 1                 # Carrier signal Amplitude
f_c = 20              # Carrier signal frequency (Hz)

# Modulating signal m(t)
A_m = A_c               # Modulating signal amplitude
f_m = 2               # Modulating signal frequency (Hz)
m1 = (A_m * np.sin(2 * np.pi * f_m * t) + 1) / 2                 # Sine wave      
m2 = (np.sign(np.sin(2 * np.pi * f_m * t)) + 1) / 2              # Square wave

# Frequency Modulated (FM) signal s(t) for sine wave 
delta_f = 15            # Frequency deviation (Hz)
s1 = (A_m * np.cos(2 * np.pi * f_c * t + delta_f * m1) + 1) / 2

# Amplitude Modulated (AM) signal s(t) for square wave   
s2 = (abs(m2) * np.cos(2 * np.pi * f_c * t) + 1) / 2

# Plotting the signals
plt.figure(figsize=(20, 20))

# carrier signal
carrier_t = (A_c * np.cos(2 * np.pi * f_c * t) + 1) / 2 
plt.subplot(3, 2, 1)
plt.plot(t, carrier_t)
plt.title('Carrier Signal (Cosine Wave)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.subplot(3, 2, 2)
plt.plot(t, carrier_t)
plt.title('Carrier Signal (Cosine Wave)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

# modulating signals
plt.subplot(3, 2, 3)
plt.plot(t, m1)
plt.title('Modulating Signal m(t) - sine wave')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.subplot(3, 2, 4)
plt.plot(t, m2)
plt.title('Modulating Signal m(t) - square wave')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

# modulated signals
plt.subplot(3, 2, 5)
plt.plot(t, s1)
plt.title('Frequency Modulated Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.subplot(3, 2, 6)
plt.plot(t, s2)
plt.title('Amplitude Modulated Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

# plt.tight_layout()
plt.subplots_adjust(hspace=0.5)
plt.show()