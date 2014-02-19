"""
Contains the functions to create trajectories plots and videos of given simulations.
Can be run directly by calling

>>> python graphics.py

with the desired simulation set in :py:data:`.constants.currentsim`.
If you don't want the graphical output for *all* of these simulations, you can *filter*
them by setting conditions with :py:func:`.utils.applyFilter` as shown in the source of :py:func:`main`.
"""


import os, string, sys
import numpy as np
import matplotlib.lines as lines
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mcclib.classMesenchymal import Mesenchymal 
from mcclib.classMaze import Maze, densityArray
import mcclib.classDataset as classDataset
from mcclib.classDataset import Dataset
import mcclib.utils as utils
import mcclib.constants as constants

def main(name):
    utils.setup_logging_base() 
    
    const = utils.readConst(constants.simdir, name)
    if const is None:
        utils.info("Could not read %s, are you sure it exists?" % (name,) )
        utils.info("Exiting.")
        sys.exit(1)
    constlist = utils.unravel(const)
    
    #If you have a sim-file causing a large number of simulations, but you want to
    #create graphics for only a few of them, you can use utils.applyFilter() as in this example:
    #constlist = utils.applyFilter(constlist, "r", [0])
    
    funcs = [
             create_path_plot,
             create_video
             ]
    if len(constlist)==0:
        utils.info("No simulations fit the filter criteria given.")
    for myconst in constlist:
        if not utils.retainCompleteDataset(myconst):
            utils.info("Cannot create graphics because the simulation %s did not retain the complete dataset. Check (sim/%s)." % (name, name) )
        else:
            utils.info("Creating graphics for %s" % (myconst["name"],) )
            writeFrames(myconst, output_func=funcs)

def create_path_plot(myconst, path, savedir, dataset, step, finalmaze=None):
    dsA = dataset
    mazepath = os.path.join(path, myconst["maze"])
    myMaze = Maze(mazepath, myconst["fieldlimits"], myconst["border"], useWeave=constants.useWeave)
    Mesenchymal.classInit(myconst, basepath=path)    
    field = myconst["fieldlimits"]
    dt = myconst["dt"]
    density = densityArray(myMaze.data.shape, field)
    
    isMesenchymal = dsA.types==Dataset.is_mesenchymal
    
    times = dsA.times
    if finalmaze is None:
        if dsA.eating.any():
            for t in xrange(times.size):
                for i in range(dsA.N_agents):
                    if dsA.eating[t,i]:
                        myMaze.degrade(dsA.positions[t,i], Mesenchymal.eat, dsA.energies[t,i], density, dt)
            myMaze.buildBorders(myMaze.data)
            myMaze.updateAll()
        finalmaze = myMaze.getMaze()
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.axis(field)
    for i in reversed(range(dsA.N_agents)):
        xpos = dsA.positions[::constants.line_resolution_step,i,0]
        ypos = dsA.positions[::constants.line_resolution_step,i,1]
        c = 'g' if isMesenchymal[i] else 'b'
        A_line = lines.Line2D(xpos, ypos, color=c, alpha=0.8)
        ax.add_line(A_line)
    plt.axis(field)
    ax.set_xlabel(r'$x_1$')
    ax.set_ylabel(r'$x_2$')
    plt.imshow(finalmaze.T, extent=field, cmap=cm.get_cmap("binary"), interpolation='nearest', alpha=1, origin='lower')
    filename = os.path.join(savedir, "paths.png")
    plt.savefig(filename, format="png")
    plt.close()

def create_video(myconst, path, resultspath, dataset, step, finalmaze=None):
    if dataset is None:
        utils.info("dataset is None. Are you sure all the simulation data was saved? Check the relevant file in \"sim/\" for \"save\_finalstats\_only\"")
        return None
    dsA = dataset
    mazepath = os.path.join(path, myconst["maze"])
    myMaze = Maze(mazepath, myconst["fieldlimits"], myconst["border"], useWeave=constants.useWeave)
    Mesenchymal.classInit(myconst, basepath=path)    
    field = myconst["fieldlimits"]
    dt = myconst["dt"]
    density = densityArray(myMaze.data.shape, field)
    
    framesdir = os.path.join(resultspath, constants.framesdir)
    utils.ensure_dir(framesdir)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    k = 0
    each_nth_image = 10
    count = 0
    times = dsA.times
    endtime = times[-1]
    for t in xrange(times.size):
        for i in xrange(dsA.N_agents):
            if dsA.eating[t,i]:
                myMaze.degrade(dsA.positions[t,i], Mesenchymal.eat, dsA.energies[t,i], density, dt)
        update_positions=dsA.positions[t, dsA.eating[t]]
        if len(update_positions)>0:
            myMaze.update(update_positions, Mesenchymal.eat, density, needGradient=True)
            myMaze.buildBorders(myMaze.data)
        if count%step==0:
            img = myMaze.getMaze()
            plotFrame(plt, ax, dsA, img, t, field, framesdir, k)
            if k%each_nth_image==0:
                utils.info("Time %s out of %s, image %s" % (round(times[t], 4), round(endtime, 4), k))
            k += 1
        count += 1
    write_mpg(resultspath, framesdir, "%s.mpg" % myconst["name"])

def plotFrame(plt, ax, dsA, maze, t, field, savedir, k):
    isAmoeboid = dsA.types==dsA.is_amoeboid
    isMesenchymal = dsA.types==dsA.is_mesenchymal 
    plt.cla()

    colors = np.array(len(isMesenchymal) * ['k'])
    colors[isMesenchymal] = 'g'
    colors[isAmoeboid] = 'b'

    plt.scatter(dsA.positions[t,isAmoeboid,0], dsA.positions[t,isAmoeboid,1], color=colors[isAmoeboid], marker='.', hold='on')
    plt.scatter(dsA.positions[t,isMesenchymal,0], dsA.positions[t,isMesenchymal,1], color=colors[isMesenchymal], marker='.')
    plt.axis(field)
    ax.set_xlabel(r'$x$', fontsize=20)
    ax.set_ylabel(r'$y$', fontsize=20)
    ax.set_title('Positions of agents at t=%s' % dsA.times[t])
    plt.imshow(maze.T, extent=field, cmap=cm.get_cmap("binary"), interpolation='bilinear', alpha=1, origin='lower')
    filename = os.path.join(savedir, 'frame_'+ string.zfill(str(k),5) +".png")
    plt.savefig(filename, format="png")

def writeFrames(myconst, output_func=[create_video]):
    path = os.getcwd() + "/"
    resultsdir = os.path.join(path, constants.resultspath, myconst["name"])
    if os.path.exists(resultsdir)==False:
        print "Directory %s does not exist. Abandoning." % resultsdir
        return
    savedir = os.path.join(path, resultsdir)
    logfilename = os.path.join(savedir, "logfile_create_video.txt")
    logging_filehandler = utils.setup_logging_sim(logfilename)
    dsA = classDataset.load(Dataset.ARRAYS, savedir, dt=myconst["dt"], fileprefix="A")
    if dsA is None:
        utils.error("Could not open dataset. Exiting.")
        return
    simfps = 1/myconst["dt"]
    vidfps = myconst["fps"]
    step = int(simfps/vidfps)
    length = len(dsA.times)
    if length<step or step==0:
        step = 1
    framesdir = os.path.join(savedir, constants.framesdir)
    with open(os.path.join(savedir, constants.finalmaze_filename), "rb") as finalmazefile:
        try:
            myfinalmaze = np.load(finalmazefile)
        except:
            myfinalmaze = None
    utils.ensure_dir(framesdir)
    utils.removeFilesByEnding(framesdir, ".png")
    for f in output_func:
        f(myconst, path, savedir, dsA, step, finalmaze=None)
    os.chdir(path)
    utils.info("Done.")
    utils.remove_logging_sim(logging_filehandler)
    
def write_mpg(resultspath, framesdir, videoname):
    pattern = 'frame_%05d.png'
    framespattern = os.path.join(resultspath,  framesdir, pattern)
    videopath = os.path.join(resultspath, videoname)
    import platform
    if platform.system()=='Windows':
        cmdPath = os.path.join(os.getcwd(), 'tools', 'ffmpeg-1.2-win32', 'bin', 'ffmpeg.exe')
        command = (cmdPath, '-y', '-i', framespattern, videopath)
        utils.exec_silent_command(command)
    else:
        command = ('mencoder',
                   'mf://%s' % os.path.join(framesdir, "frame_*.png"),
                   '-mf',
                   'type=png:fps=10',
                   '-ovc',
                   'lavc',
                   '-lavcopts',
                   'vcodec=mpeg4',
                   '-oac',
                   'copy',
                   '-o',
                   os.path.join(resultspath, videoname))
        utils.exec_silent_command(command)

if __name__ == '__main__':
    main(constants.currentsim)
