#include <errno.h>
#include <fcntl.h>
#include <linux/videodev2.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <time.h>
#include <unistd.h>

static double now_seconds(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return value.tv_sec + value.tv_nsec / 1e9;
}

static int request(int descriptor, unsigned long operation, void *argument) {
    int result;
    do { result = ioctl(descriptor, operation, argument); } while (result < 0 && errno == EINTR);
    if (result < 0) perror("V4L2 ioctl");
    return result;
}

int main(int count, char **arguments) {
    if (count < 3) return 2;
    int descriptor = open(arguments[1], O_RDWR | O_NONBLOCK);
    if (descriptor < 0) { perror("open camera"); return 3; }
    struct v4l2_format format = { .type = V4L2_BUF_TYPE_VIDEO_CAPTURE };
    format.fmt.pix.width = count > 3 ? atoi(arguments[3]) : 640;
    format.fmt.pix.height = count > 4 ? atoi(arguments[4]) : 480;
    format.fmt.pix.pixelformat = v4l2_fourcc(arguments[2][0],arguments[2][1],arguments[2][2],arguments[2][3]);
    format.fmt.pix.field = V4L2_FIELD_ANY;
    if (request(descriptor,VIDIOC_S_FMT,&format) < 0) return 4;
    struct v4l2_streamparm rate = { .type = V4L2_BUF_TYPE_VIDEO_CAPTURE };
    rate.parm.capture.timeperframe.numerator = 1;
    rate.parm.capture.timeperframe.denominator = 30;
    if (request(descriptor,VIDIOC_S_PARM,&rate) < 0) return 5;
    if (request(descriptor,VIDIOC_G_FMT,&format) < 0 || request(descriptor,VIDIOC_G_PARM,&rate) < 0) return 6;
    if (count > 5) {
        struct v4l2_control manual = { .id=V4L2_CID_EXPOSURE_AUTO, .value=1 };
        struct v4l2_control exposure = { .id=V4L2_CID_EXPOSURE_ABSOLUTE, .value=atoi(arguments[5]) };
        struct v4l2_control power = { .id=V4L2_CID_POWER_LINE_FREQUENCY, .value=0 };
        if(request(descriptor,VIDIOC_S_CTRL,&manual)<0 || request(descriptor,VIDIOC_S_CTRL,&power)<0 || request(descriptor,VIDIOC_S_CTRL,&exposure)<0) return 14;
    }
    unsigned int code = format.fmt.pix.pixelformat;
    printf("{\"kind\":\"mode\",\"width\":%u,\"height\":%u,\"fourcc\":\"%c%c%c%c\",\"interval_numerator\":%u,\"interval_denominator\":%u}\n",
        format.fmt.pix.width,format.fmt.pix.height,code&255,(code>>8)&255,(code>>16)&255,(code>>24)&255,
        rate.parm.capture.timeperframe.numerator,rate.parm.capture.timeperframe.denominator);
    struct v4l2_requestbuffers buffers = { .count = 4, .type = V4L2_BUF_TYPE_VIDEO_CAPTURE, .memory = V4L2_MEMORY_MMAP };
    if (request(descriptor,VIDIOC_REQBUFS,&buffers) < 0 || buffers.count > 32 || buffers.count == 0) return 7;
    void *mapped[32] = {0};
    size_t lengths[32] = {0};
    for (unsigned int index = 0; index < buffers.count; index++) {
        struct v4l2_buffer buffer = { .type = V4L2_BUF_TYPE_VIDEO_CAPTURE, .memory = V4L2_MEMORY_MMAP, .index = index };
        if (request(descriptor,VIDIOC_QUERYBUF,&buffer) < 0) return 8;
        lengths[index] = buffer.length;
        mapped[index] = mmap(NULL,buffer.length,PROT_READ|PROT_WRITE,MAP_SHARED,descriptor,buffer.m.offset);
        if (mapped[index] == MAP_FAILED || request(descriptor,VIDIOC_QBUF,&buffer) < 0) return 9;
    }
    enum v4l2_buf_type stream = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (request(descriptor,VIDIOC_STREAMON,&stream) < 0) return 10;
    double started = now_seconds();
    int failure = 0;
    int controls_read=0;
    while (now_seconds() - started < 15.0) {
        fd_set ready;
        FD_ZERO(&ready); FD_SET(descriptor,&ready);
        struct timeval timeout = { .tv_sec = 2 };
        int selected = select(descriptor+1,&ready,NULL,NULL,&timeout);
        if (selected < 0 && errno == EINTR) continue;
        if (selected <= 0) { failure = 11; break; }
        struct v4l2_buffer buffer = { .type = V4L2_BUF_TYPE_VIDEO_CAPTURE, .memory = V4L2_MEMORY_MMAP };
        if (request(descriptor,VIDIOC_DQBUF,&buffer) < 0) { failure = 12; break; }
        double received = now_seconds();
        if (!controls_read && received-started>4.0) {
            unsigned int ids[]={V4L2_CID_EXPOSURE_AUTO,V4L2_CID_EXPOSURE_ABSOLUTE,V4L2_CID_POWER_LINE_FREQUENCY};
            for(int index=0;index<3;index++) {
                struct v4l2_control setting={.id=ids[index]};
                if(request(descriptor,VIDIOC_G_CTRL,&setting)==0) printf("{\"kind\":\"control\",\"id\":%u,\"value\":%d}\n",setting.id,setting.value);
            }
            controls_read=1;
        }
        if (received - started > 4.0)
            printf("{\"kind\":\"frame\",\"sequence\":%u,\"bytesused\":%u,\"flags\":%u,\"error\":%s,\"driver_timestamp\":%.9f,\"host_timestamp\":%.9f}\n",
                buffer.sequence,buffer.bytesused,buffer.flags,(buffer.flags&V4L2_BUF_FLAG_ERROR)?"true":"false",
                buffer.timestamp.tv_sec+buffer.timestamp.tv_usec/1e6,received);
        if (request(descriptor,VIDIOC_QBUF,&buffer) < 0) { failure = 13; break; }
    }
    request(descriptor,VIDIOC_STREAMOFF,&stream);
    for (unsigned int index = 0; index < buffers.count; index++) munmap(mapped[index],lengths[index]);
    close(descriptor);
    return failure;
}
