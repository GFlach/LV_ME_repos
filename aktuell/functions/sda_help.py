#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 17:01:41 2017
Hilfsfunktionen für Datenanalyse (TSV)
@author: gudrun
"""

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import random as rd
import pandas as pd

#from random import randint


def plot_data(dr, hist, min_d, max_d):
    plt.figure(figsize=(12,5))
    plt.subplot(121)
    plt.plot(dr, 'b.')
    plt.title('Datenreihe')
    plt.xlabel('Index')
    plt.ylabel('Wert')
    plt.subplot(122)
    x = np.arange(min_d, max_d, 1)
    plt.stem(x, hist, markerfmt='.')
    #plt.xticks(np.arange(min_d, max_d + 10, 10))
    plt.title('Histogramm')
    plt.xlabel('Wert')
    plt.ylabel('Häufigkeit')
    plt.show()
    
# Gleichverteilung
def randnum(anz, zmin, zmax, mu, sigma, mode, alpha, beta, lamd, p_alpha):
    zz = np.zeros(anz)
    for i in range(0, anz):
        zz[i] = rd.uniform(zmin, zmax)
        #zz[i] = randint(min_z, max_z)
    mybins = np.arange(zmin + 1, zmax + 2, 1)
    zzhist, zzbins = np.histogram(zz, mybins)
    return(zz, zzhist)

# Normalverteilung
def norm(anz, zmin, zmax, mu, sigma, mode, alpha, beta, lamd, p_alpha):
    zz = np.zeros(anz)
    for i in range(0, anz):
        zz[i] = rd.normalvariate(mu, sigma)
    mybins = np.arange(zmin, zmax + 1, 1)
    zzhist, zzbins = np.histogram(zz, mybins)
    return(zz, zzhist)

#Dreieckverteilung
def triangular(anz, zmin, zmax, mu, sigma, mode, alpha, beta, lamd, p_alpha):
    zz = np.zeros(anz)
    for i in range(0, anz):
        zz[i] = rd.triangular(zmin,zmax, mode)
    mybins = np.arange(zmin, zmax + 1, 1)
    zzhist, zzbins = np.histogram(zz, mybins)
    return(zz, zzhist)

# Betaverteilung
def betadist(anz, zmin, zmax, mu, sigma, mode, alpha, beta, lamd, p_alpha):
    zz = np.zeros(anz)
    for i in range(0, anz):
        zz[i] = rd.betavariate(alpha, beta)*(zmax + abs(zmin)) - abs(zmin)
        mybins = np.arange(zmin, zmax + 1, 1)
    zzhist, zzbins = np.histogram(zz, mybins)
    return(zz, zzhist)

# Gammaverteilung
def gammadist(anz, zmin, zmax, mu, sigma, mode, alpha, beta, lamd, p_alpha):
    zz = np.zeros(anz)
    for i in range(0, anz):
        zz[i] = rd.gammavariate(alpha, beta)*(zmax + abs(zmin)) - abs(zmin)
    mybins = np.arange(zmin, zmax + 1, 1)
    zzhist, zzbins = np.histogram(zz, mybins)
    return(zz, zzhist)

# Exponentialverteilung
def expodist(anz, zmin, zmax, mu, sigma, mode, alpha, beta, lamd, p_alpha):
    zz = np.zeros(anz)
    for i in range(0, anz):
        zz[i] = rd.expovariate(lamd)*(zmax + abs(zmin)) - abs(zmin)
    mybins = np.arange(zmin, zmax + 1, 1)
    zzhist, zzbins = np.histogram(zz, mybins)
    return(zz, zzhist)

# Paretoverteilung
def paretodist(anz, zmin, zmax, mu, sigma, mode, alpha, beta, lamd, p_alpha):
    zz = np.zeros(anz)
    for i in range(0, anz):
        #zz[i] = int(rd.paretovariate(alpha)*(max_z + abs(min_z)) - abs(min_z))
        zz[i] = zmin - 1 + int(rd.paretovariate(alpha))
    mybins = np.arange(zmin, zmax + 1, 1)
    zzhist, zzbins = np.histogram(zz, mybins)
    return(zz, zzhist)

def test_vt(anz, zmin, zmax, mu, sigma, mode, alpha, beta, lamd, p_alpha, vt, save=False, name='xxx.jpg', nv=True):
    if vt == 'randnum':
        zz, zzhist = randnum(anz, zmin, zmax, mu, sigma, mode, alpha, beta, lamd, p_alpha)
        mu_r = np.mean(zz)
        sig_r = np.sqrt(np.var(zz))
        print(print_info(vt, anz, zmin, zmax, mu, sigma, mu_r, sig_r))
        x = np.arange(zmin, zmax, 1)
        plt.stem(x,zzhist, markerfmt = '.')
        if nv == True:
            y = 1/(sig_r * np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu_r)/sig_r)**2)*anz
            plt.plot(x, y, 'r')
    if vt == 'norm':
        zz, zzhist = norm(anz, zmin, zmax, mu, sigma, mode, alpha, beta, lamd, p_alpha)
        mu_r = np.mean(zz)
        sig_r = np.sqrt(np.var(zz))
        print(print_info(vt, anz, zmin, zmax, mu, sigma, mu_r, sig_r))
        x = np.arange(zmin, zmax, 1)
        plt.stem(x,zzhist, markerfmt = '.')
        if nv == True:
            y = 1/(sig_r * np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu_r)/sig_r)**2)*anz
            plt.plot(x, y, 'r')
    if vt == 'triangle':
        zz, zzhist = triangular(anz, zmin, zmax, mu, sigma, mode, alpha, beta, lamd, p_alpha)
        mu_r = np.mean(zz)
        sig_r = np.sqrt(np.var(zz))
        print(print_info(vt, anz, zmin, zmax, mu, sigma, mu_r, sig_r))
        x = np.arange(zmin, zmax, 1)
        plt.stem(x,zzhist, markerfmt = '.')
        if nv == True:
            y = 1/(sig_r * np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu_r)/sig_r)**2)*anz
            plt.plot(x, y, 'r')
    if vt == 'beta':
        zz, zzhist = betadist(anz, zmin, zmax, mu, sigma, mode, alpha, beta, lamd, p_alpha)
        mu_r = np.mean(zz)
        sig_r = np.sqrt(np.var(zz))
        print(print_info(vt, anz, zmin, zmax, mu, sigma, mu_r, sig_r))
        x = np.arange(zmin, zmax, 1)
        plt.stem(x,zzhist, markerfmt = '.')
        if nv == True:
            y = 1/(sig_r * np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu_r)/sig_r)**2)*anz
            plt.plot(x, y, 'r')
    if vt == 'gamma':
        zz, zzhist = gammadist(anz, zmin, zmax, mu, sigma, mode, alpha, beta, lamd, p_alpha)
        mu_r = np.mean(zz)
        sig_r = np.sqrt(np.var(zz))
        print(print_info(vt, anz, zmin, zmax, mu, sigma, mu_r, sig_r))
        x = np.arange(zmin, zmax, 1)
        plt.stem(x,zzhist, markerfmt = '.')
        if nv == True:
            y = 1/(sig_r * np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu_r)/sig_r)**2)*anz
            plt.plot(x, y, 'r')
    if vt == 'expo':
        zz, zzhist = expodist(anz, zmin, zmax, mu, sigma, mode, alpha, beta, lamd, p_alpha)
        mu_r = np.mean(zz)
        sig_r = np.sqrt(np.var(zz))
        print(print_info(vt, anz, zmin, zmax, mu, sigma, mu_r, sig_r))
        x = np.arange(zmin, zmax, 1)
        plt.stem(x,zzhist, markerfmt = '.')
        if nv == True:
            y = 1/(sig_r * np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu_r)/sig_r)**2)*anz
            plt.plot(x, y, 'r')
    if vt == 'pareto':
        zz, zzhist = paretodist(anz, zmin, zmax, mu, sigma, mode, alpha, beta, lamd, p_alpha)
        mu_r = np.mean(zz)
        sig_r = np.sqrt(np.var(zz))
        print(print_info(vt, anz, zmin, zmax, mu, sigma, mu_r, sig_r))
        x = np.arange(zmin, zmax, 1)
        plt.stem(x,zzhist, markerfmt = '.')
        if nv == True:
            y = 1/(sig_r * np.sqrt(2*np.pi))*np.exp(-0.5*((x-mu_r)/sig_r)**2)*anz
            plt.plot(x, y, 'r')
        #plt.show()
    if save == True:
        plt.savefig(name)
        
def print_info(vt, anz, zmin, zmax, mu, sigma, mu_r, sig_r):
    string0 = 'Verteilung: ' + vt + '\n'
    string1 = 'Anzahl Werte: ' + str(anz) + '\n'
    info = string0 + string1
    if vt == 'pareto':
        string2 = 'kleinster Wert: ' + str(zmin) + '\n'
        info = info + string2
    if vt in ('randnum', 'triangle', 'gamma', 'beta'):
        string2 = 'kleinster Wert: ' + str(zmin) + '\n'
        string3 = 'größter Wert: ' + str(zmax) + '\n'
        info = info + string2 + string3
    if vt == 'norm':
        string4 = 'Mittelwert(vorgegeben): ' + str(mu) + '\n'
        string5 = 'Standardabweichung(vorgegeben): ' + str(sigma) + '\n'
        info = info + string4 + string5
    string6 = 'Mittelwert(berechnet): ' + str(mu_r) + '\n'
    string7 = 'Standardabweichung(berechnet): ' + str(sig_r) + '\n'
    info = info + string6 + string7
    return(info)

def generate_data(anz, zmin, zmax, mu, sigma, mode, alpha, beta, lamd, p_alpha, vt):
    if vt == 'randnum':
        data, hist = randnum(anz, zmin, zmax, mu, sigma, mode, alpha, beta, lamd, p_alpha)
    if vt == 'norm':
        data, hist = norm(anz, zmin, zmax, mu, sigma, mode, alpha, beta, lamd, p_alpha)
    if vt == 'triangle':
        data, hist = triangular(anz, zmin, zmax, mu, sigma, mode, alpha, beta, lamd, p_alpha)
    if vt == 'beta':
        data, hist = betadist(anz, zmin, zmax, mu, sigma, mode, alpha, beta, lamd, p_alpha)
    if vt == 'gamma':
        data, hist = gammadist(anz, zmin, zmax, mu, sigma, mode, alpha, beta, lamd, p_alpha)
    if vt == 'expo':
        data, hist = expodist(anz, zmin, zmax, mu, sigma, mode, alpha, beta, lamd, p_alpha)
    if vt == 'pareto':
        data, hist = paretodist(anz, zmin, zmax, mu, sigma, mode, alpha, beta, lamd, p_alpha)
    return(data)
    
def auto_generate(anz, zmin, zmax, mu, sigma, mode, alpha, beta, lamd, p_alpha):
    info = []
    d = generate_data(anz, zmin, zmax, mu, sigma, mode, alpha, beta, lamd, p_alpha, 'randnum')
    dmu = np.round(np.mean(d),2)
    sig = np.round(np.sqrt(np.var(d)),2)
    info.append(['randnum', anz, zmin, zmax, '-', '-', dmu, sig, '-', '-', '-', '-', '-'])
        
    d = generate_data(anz, zmin, zmax, mu, sigma, mode, alpha, beta, lamd, p_alpha, 'norm')
    dmu = np.round(np.mean(d),2)
    sig = np.round(np.sqrt(np.var(d)),2)
    info.append(['norm', anz, '-', '-', mu, sigma, dmu, sig, '-', '-', '-', '-', '-'])
    
    d = generate_data(anz, zmin, zmax, mu, sigma, mode, alpha, beta, lamd, p_alpha, 'triangle')
    dmu = np.round(np.mean(d),2)
    sig = np.round(np.sqrt(np.var(d)),2)
    info.append(['triangular', anz, zmin, zmax, '-', '-', dmu, sig, mode, '-', '-', '-', '-'])

    d = generate_data(anz, zmin, zmax, mu, sigma, mode, alpha, beta, lamd, p_alpha, 'beta')
    dmu = np.round(np.mean(d),2)
    sig = np.round(np.sqrt(np.var(d)),2)
    info.append(['betadist', anz, zmin, zmax, '-', '-', dmu, sig, '-', alpha, beta, '-', '-'])

    d = generate_data(anz, zmin, zmax, mu, sigma, mode, alpha, beta, lamd, p_alpha, 'gamma')
    dmu = np.round(np.mean(d),2)
    sig = np.round(np.sqrt(np.var(d)),2)
    info.append(['gammadist', anz, zmin, '-', '-', '-', dmu, sig, '-', alpha, beta, '-', '-'])

    d = generate_data(anz, zmin, zmax, mu, sigma, mode, alpha, beta, lamd, p_alpha, 'expo')
    dmu = np.round(np.mean(d),2)
    sig = np.round(np.sqrt(np.var(d)),2)
    info.append(['expodist', anz, zmin, '-', '-', '-', dmu, sig, '-', '-', '-', lamd, '-'])

    d = generate_data(anz, zmin, zmax, mu, sigma, mode, alpha, beta, lamd, p_alpha, 'pareto')
    dmu = np.round(np.mean(d),2)
    sig = np.round(np.sqrt(np.var(d)),2)
    info.append(['paretodist', anz, zmin, '-', '-', '-', dmu, sig, '-', '-', '-', '-', p_alpha])
    return(info)
    
def build_df(data_files, class_labels):
    i = 0
    for fn in data_files:
        dname = 'data/' + fn + '.npy'
        d = np.load(dname)
        df = pd.DataFrame(np.transpose(d), columns=['x1', 'x2'])
        kn = [class_labels[i]]
        for j in range(0,d.shape[1]-1):
            kn = kn + [class_labels[i]]
        dfk = pd.DataFrame(np.transpose(kn), columns=['Klasse'])
        if i == 0:
            df_ges = df
            df_cn = dfk
        else:
            df_ges = df_ges.append(df)
            df_cn = df_cn.append(dfk)
        i = i + 1
    df_ges = pd.concat([df_ges, df_cn], axis = 1)  
    return(df_ges)

def show_res(X, y, w):
    pmin = min(min(X[:,1]), min(X[:,2]))*1.1
    pmax = max(max(X[:,1]), max(X[:,2]))*1.1
    colors = ['green', 'red', 'blue', 'yellow', 'magenta', 'lime', 'skyblue', 'orangered', 'cyan', 'aqua']
    hist_cl = np.unique(y)
    eps = 10**(-6)
    plt.gcf().clear()
    plt.figure(figsize=(7,7))
    for hist_cl, color in zip(hist_cl, colors):
        X1 = X[np.array([index for index, val in enumerate(y) if val == hist_cl])]
        plt.scatter(X1[:,1], X1[:,2], color=color, marker='o', label='Klasse ' + str(hist_cl))
    p1 = -w[1]/(w[2] + eps)*pmin -w[0]/(w[2] + eps)
    p2 = -w[1]/(w[2] + eps)*pmax -w[0]/(w[2] + eps)
    plt.plot([pmin, pmax], [p1, p2])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(loc='upper left')
    plt.axis([pmin, pmax, pmin, pmax])
    plt.grid()
    plt.show()
    
def show_res1(df_ges, w):
    colors = ['green', 'red', 'blue', 'yellow', 'magenta', 'lime', 'skyblue', 'orangered', 'cyan', 'aqua']
    hist_cl = np.unique(df_ges['Klasse'].values)
    eps = 10**(-6)
    plt.figure(figsize=(7,7))
    for hist_cl, color in zip(hist_cl, colors):
        df_h = df_ges[df_ges.Klasse == hist_cl].values
        y = df_h[:,0:2]
        plt.scatter(y[:,0], y[:,1], color=color, marker='o', label='Klasse ' + str(hist_cl))
    p1 = w[1]/(w[2] + eps) -w[0]/(w[2] + eps)
    p2 = -w[1]/(w[2] + eps)*120 -w[0]/(w[2] + eps)
    plt.plot([0, 120], [p1, p2])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend(loc='upper left')
    plt.axis([0, 120, 0, 120])
    plt.grid()
    plt.show()

def perceptron(X, y, w, alpha, n_iter):
    for i in range(0, n_iter):
        err = 0
        for d in range(0, len(X)):
            e = np.sign(np.dot(X[d], w)) - y[d]
            if e != 0:
                err = err + 1
                w = w + 2 * alpha * X[d] * y[d]
        if err == 0:
            break
    return(i, w)    

def plot_decision_regions(X, y, w, resolution=0.1):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    #Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = np.dot(w[1:,],np.array([xx1.ravel(), xx2.ravel()]))
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')
