# necessary imports
#Most of the data pre-processing was done in the python notebook. 
#this file contains only the necessary data and files to run the frontend
import pandas as pd
import numpy as np
#these imports are necessary to run the recommendation model. Used for calculating similarities
import sklearn.metrics.pairwise as pw
import scipy.spatial.distance as scpy


#the pre-processed data is stored in the file 'sentiDataFinal.csv'
sentiData = pd.read_csv('sentiDataFinal.csv')

#normalizing the weightage given to the year by subtracting the mean from the data and dividing by standard deviation.
sMean = sentiData['year'].mean()
sStdev = sentiData['year'].std()
yearNorm = sentiData.year.apply(lambda x: (x - sMean) / sStdev)
sentiData['yearNorm'] = yearNorm


#this function calculates the eucledean distance between the two given songs
def get_eucDist(i, j):
    # Year, the 2 sentiment scores and genre have been considered as vectors for this function.
    vec1 = [sentiData['yearNorm'][i] / 100, sentiData['Sentiment'][i], sentiData['Sentiment2'][i]]
    vec2 = [sentiData['yearNorm'][j] / 100, sentiData['Sentiment'][j], sentiData['Sentiment2'][j]]

    genreWt = 1 / 100  # normalization factor found by trial and error

    # We are simulating a kind of OH encoding over here
    if sentiData['genre'][i] == sentiData['genre'][j]:
        vec1.append(genreWt)
        vec2.append(genreWt)
    else:
        vec1.append(genreWt)
        vec2.append(0)

    return scpy.euclidean(vec1, vec2)


Eucd_data = sentiData.copy()

#running a loop to find the Euclidean distance between the given song and all other songs
def get_dist_songs(i):
    dist = Eucd_data.apply(lambda row: get_eucDist(i, row.name), axis=1)
    Eucd_data['Dist'] = dist
    Eucd_data.sort_values(by='Dist', ascending=True, inplace=True)


#getting cosine similarity of the data, similar to eucledean distance 
def get_cosSim(i, j, sliderVal, wt):
    vec1 = [sentiData['yearNorm'][i] / 100, sentiData['Sentiment'][i], sentiData['Sentiment2'][i]]
    vec2 = [sentiData['yearNorm'][j] / 100, sentiData['Sentiment'][j], sentiData['Sentiment2'][j]]
    # simulating OH encoding for artist and genre without actually doing it
    # The artist weight is normalised by a process similar it TF-IDF, 
    # where wt is the square root of the frequency of the artist(ie, number of songs he has played)
    # the slider value is given so that the User can adjust if he wants the weightage of the song to be high or not
    if sliderVal == 0:
        artistWt = 100
    else:
        artistWt = 0.8 * (1 - sliderVal) / wt  # found by trial and error, see old tests
    genreWt = 0.001
    if sentiData['artist'][i] == sentiData['artist'][j]:
        vec1.append(artistWt)
        vec2.append(artistWt)
    else:
        vec1.append(artistWt)
        vec2.append(0)
    #encoding of genre similar to that in the euclidean distance
    if sentiData['genre'][i] == sentiData['genre'][j]:
        vec1.append(genreWt)
        vec2.append(genreWt)
    else:
        vec1.append(genreWt)
        vec2.append(0)
    return pw.cosine_similarity([vec1], [vec2])[0][0]

# gets the cosine similarity for the top 10% closest songs(eucledean) to a given song
def get_nearest_neighbor(i, sliderVal):
    # important: testerData is a local variable, does not effect the outside testerData
    # that's why we return the dataframe
    testerData = Eucd_data[:(len(sentiData) // 10)].copy()
    #groups of  similar genres are created, so that genres with very few songs are not overshadowed.
    genre = sentiData.iloc[i]['genre']
    if genre in ['Jazz', 'R&B', 'Indie', 'Electronic']:
        testerData = testerData[testerData['genre'] == genre]
    elif genre in ['Pop', 'Hip-Hop']:
        testerData = testerData[testerData['genre'].isin(['Pop', 'Hip-Hop', 'Other'])]
    elif genre in ['Rock', 'Metal']:
        testerData = testerData[testerData['genre'].isin(['Rock', 'Metal', 'Other'])]
    elif genre in ['Country', 'Folk']:
        testerData = testerData[testerData['genre'].isin(['Country', 'Folk', 'Other'])]
    #weightage for the song normalization
    wt = len(sentiData[sentiData.artist == sentiData['artist'][i]]) ** 0.5
    
    #mosel calculation
    score = testerData.apply(lambda row: get_cosSim(i, row.name, sliderVal, wt), axis=1)
    testerData['Score'] = score
    testerData.sort_values(by='Score', ascending=False, inplace=True)
    return testerData

#function to handle cases if the user does not type the complete name of artist or song
#it predicts possible songs the user may be looking for.
def search_list_predict(term_list, search_term):
    filtered_list = [x for x in term_list if x.startswith(search_term)]
    filtered_list.extend([x for x in term_list if (search_term in x) and (x not in filtered_list)])
    filtered_set = set(filtered_list)
    filtered_list = list(filtered_set)
    return filtered_list


# Search function to search the song with name of song and artist with artist name optional
def get_info(songName, artistName=None):
    #case when the user does not enter artist name
    if artistName is None:
        #if exact match of song is found
        if len(sentiData[sentiData['song'] == songName]):
            return sentiData[sentiData['song'] == songName]
        else:
            #there is no exact match of the song, so it tries to see if there could be other possibilities
            #if there are no other possibilities, an error message is returned.
            possible_song_names = search_list_predict(sentiData['song'], songName)
            if len(possible_song_names) == 0:
                print('Error: song not found')
                return -1
            else:
                df = pd.DataFrame()
                for x in possible_song_names:
                    df = pd.concat([df, sentiData[sentiData['song'] == x]], axis=0)
                return df
    #case when both song and artist is entered
    else:
        #if exact match is found:
        if len(sentiData[(sentiData['song'] == songName) & (sentiData['artist'] == artistName)]):
            return sentiData[(sentiData['song'] == songName) & (sentiData['artist'] == artistName)]
        #if artist name is correct but song name is incomplete, then it predicts a possible song
        elif len(sentiData[sentiData['artist'] == artistName]):
            possible_song_names = search_list_predict(sentiData['song'], songName)
            if len(possible_song_names) == 0:
                print('Error: song not found')
                return -1
            else:
                df = pd.DataFrame()
                for x in possible_song_names:
                    df = pd.concat([df, sentiData[(sentiData['song'] == x) & (sentiData['artist'] == artistName)]],
                                   axis=0)
                    if len(df) == 0:
                        print('Error: song not found')
                        return -1
                    else:
                        return df
        #if Song name matches and artist does not match, it predicts a possible artist
        elif len(sentiData[sentiData['song'] == songName]):
            possible_artist_names = search_list_predict(sentiData['artist'], artistName)
            df = pd.DataFrame()
            for y in possible_artist_names:
                df = pd.concat([df, sentiData[(sentiData['song'] == songName) & (sentiData['artist'] == y)]], axis=0)
            if len(df) == 0:
                return sentiData[sentiData['song'] == songName]
            else:
                return df
        #if both dont match, it predicts both
        else:
            possible_song_names = search_list_predict(sentiData['song'], songName)
            possible_artist_names = search_list_predict(sentiData['artist'], artistName)
            if len(possible_song_names) == 0:
                print('Error: song not found')
                return -1
            else:
                df = pd.DataFrame()
                for x in possible_song_names:
                    for y in possible_artist_names:
                        df = pd.concat([df, sentiData[(sentiData['song'] == x) & (sentiData['artist'] == y)]], axis=0)
                if len(df) == 0:
                    print('Error: song not found')
                    return -1
                else:
                    return df


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################
# frontend starts here
import tkinter as tk
import tkinter.scrolledtext as scrolledtext

#function that accepts user input, runs the search function and calculatesthe playlist by calling to the functions above.
def display_playlist(songInput, artistInput, sliderInput):
    #cleanig data
    display.delete(1.0, tk.END)
    display.insert(1.0, 'Processing, please wait...')
    songName = songInput.lower()
    artistName = artistInput.lower()
    for ch in ["'", '"', '.', ',', ';', '(', ')', '{', '}', '[', ']', ':', '!', '?']:
        if ch in songName:
            songName = songName.replace(ch, '-')
        if ch in artistName:
            artistName = artistName.replace(ch, '-')
    songName = '-'.join(songName.split())  # replaces spaces, but only those between words
    songName = '-'.join([x for x in songName.split('-') if x])  # replaces repeated '-'s, such as a--b due to "A 'b"
    artistName = '-'.join(artistName.split())
    artistName = '-'.join([x for x in artistName.split('-') if x])
    if songName == '' or songName == 'enter-a-song-name' or songName is None:
        display.delete(1.0, tk.END)
        display.insert(1.0, 'ERROR: Please input a song name!')
        return -1
    elif artistName == '' or artistName == 'enter-the-artist':
        artistName = None
    
    #calling the search algorithm and proceeding accordingly
    info = get_info(songName, artistName)
    if type(info) == int and info == -1:
        display.delete(1.0, tk.END)
        display.insert(1.0, 'ERROR: No similar songs found!')
        return -1
    info.reset_index(inplace=True)
    if len(info) > 1:
        display.delete(1.0, tk.END)
        display.insert(1.0, 'Multiple similar results found. Please select a song and artist from below and try again.\n')
        for i in range(len(info)):
            display.insert(tk.END, (str(i + 1) + '. ' + info['song'][i] + ' by ' + info['artist'][i] + '\n'))
        return 0
    else:
        display.delete(1.0, tk.END)
        display.insert(1.0, 'The songs recommended for you are:\n')
        ind = info['index'][0]
        get_dist_songs(ind)
        testPlaylist = get_nearest_neighbor(ind, sliderInput)
        testPlaylist = testPlaylist[:30]
        testPlaylist.reset_index(inplace=True)
        # shuffles songs 6-30 so there's variety and not just the same artist
        testPlaylist = pd.concat([testPlaylist[:6], testPlaylist[6:].sample(frac=1)])
        testPlaylist.reset_index(inplace=True, drop=True)
        for i in range(len(testPlaylist)):
            display.insert(tk.END,
                           (str(i + 1) + '. ' + testPlaylist['song'][i] + ' by ' + testPlaylist['artist'][i] + '\n'))
        return 0


root = tk.Tk()
root.title('MusicMoods')
bgimg = tk.PhotoImage(file="bgimg_edited.png")  # background image used for all main pages
w = bgimg.width()  # width of background image
h = bgimg.height()  # height of background image
root.geometry("%dx%d" % (w, h))  # setting size of window to that of image
root.resizable(False, False)  # prevents user from resizing main window and messing up pages

bg = tk.Label(root, image=bgimg)  # creating a background image
bg.place(x=0, y=0, relwidth=1, relheight=1)

defSongTxt = tk.StringVar(bg, value='Enter a song name')
tb1 = tk.Entry(bg, textvariable=defSongTxt, font=("arial", 10), justify=tk.CENTER, width=20)
tb1.place(x=600, y=100)

defArtTxt = tk.StringVar(bg, value='Enter the artist')
tb2 = tk.Entry(bg, textvariable=defArtTxt, font=("arial", 10), justify=tk.CENTER, width=20)
tb2.place(x=765, y=100)

divTxt = tk.Label(bg, text="Artist diversity", font=("arial", 10), width=38)
divTxt.place(x=600, y=150)
slider = tk.Scale(bg, from_=0, to=1, digits=2, resolution=0.1, orient=tk.HORIZONTAL, length=304)
slider.set(0.5)
slider.place(x=600, y=170)
bottomSliderTxt = tk.Label(bg, text="Same artist <------------------------------> Diverse Artists", font=("arial", 10),
                           width=38)
bottomSliderTxt.place(x=600, y=210)

genPlayBtn = tk.Button(bg, text="Generate Playlist!", width=38, font=("arial", 10),
                       command=lambda: display_playlist(tb1.get(), tb2.get(), slider.get()))
genPlayBtn.place(x=600, y=250)

displayTxt = 'Enter a song name and artist. \nAfter pressing the button, please allow upto a minute for processing.'
display = scrolledtext.ScrolledText(bg, width=37, height=10, undo=True, wrap='word')
display.insert(1.0, displayTxt)
display.place(x=600, y=300)

root.mainloop()
