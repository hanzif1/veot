## Instructions
<b>train.csv</b>: QA annotations are identical to <a href="https://github.com/doc-doc/NExT-QA">NExT-QA</a> train.csv, except that we slightly change the format by replacing the correct answer_id with the corresponding text answer.

<b>test.csv/val.csv</b>: QA annotations that are subset of NExT-QA test.csv/val.csv. We exclude the questions that rely on global video content and those in the descriptive group.

<b>gsub_test.json/gsub_val.json</b>: time span annotations corresponding to the QAs in test.csv/val.csv
Note：
```
{"10001787725": #video_id
    {
      "duration": 34, #Video duration (s)
      "location": #Segment
        {
          "1": [[1.2, 5.8]], #Segment corresponding to question id qid 1
          "3": [[12.1, 17.1], [20.0, 23.5], [29.7, 33.2]] #Segment corresponding to question id qid 3
          ...
        },
      fps: 29.97 #frame rate
    }
...
}
```

<b>frame2time_test.json/frame2time_val.json</b>: map the frame id into time seconds.

<b>map_vid_vidorID</b>: map the video_id in QA annotation file into video path.

<b>upbd_test.json/upbd_val.json</b>: sampled video timestamps for each video.


##################################################

train.csv: 该文件的问答注释与 NExT-QA 的 train.csv 相同，唯一的区别是我们略微改变了格式，将正确的 answer_id 替换为相应的文本答案。

test.csv/val.csv: 该文件包含的是 NExT-QA 的 test.csv 和 val.csv 的子集。我们排除了那些依赖于全局视频内容的题目以及描述性问题（descriptive group）的题目。

gsub_test.json/gsub_val.json: 该文件包含与 test.csv 和 val.csv 中的问答数据对应的时间段注释。注释的格式如下：

frame2time_test.json/frame2time_val.json: 该文件将帧 ID 映射到时间（秒）上。

map_vid_vidorID: 该文件将问答注释文件中的 video_id 映射到视频文件的路径。

upbd_test.json/upbd_val.json: 该文件包含了每个视频的采样时间戳。

