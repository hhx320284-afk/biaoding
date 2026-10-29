/* 复用工作区已有原生入口模式；项目外层只保留一个中文启动入口。
 * 重建（在本文件目录）：
 * gcc -std=c11 -O2 -Wall -Wextra -Werror 双击启动入口.c -o ../../启动手眼标定
 */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

static int show_error(const char *message)
{
    fprintf(stderr, "%s\n", message);
    pid_t dialog = fork();
    if (dialog == 0) {
        execlp("zenity", "zenity", "--error", "--no-markup",
               "--title=手眼标定启动失败", "--text", message, (char *)NULL);
        _exit(127);
    }
    if (dialog > 0) {
        while (waitpid(dialog, NULL, 0) < 0 && errno == EINTR) { }
    }
    return EXIT_FAILURE;
}

int main(int argc, char **argv)
{
    char project[PATH_MAX];
    ssize_t length = readlink("/proc/self/exe", project, sizeof(project) - 1);
    if (length < 0 || (size_t)length >= sizeof(project) - 1) {
        return show_error("无法确定启动入口的位置。请将入口保留在手眼标定项目根目录。");
    }
    project[length] = '\0';
    char *separator = strrchr(project, '/');
    if (separator == NULL) return show_error("无法确定手眼标定项目目录。");
    *separator = '\0';
    const char *script = "程序/运行支持/启动界面.sh";
    if (chdir(project) != 0 || access(script, R_OK) != 0) {
        return show_error("未找到可读取的 程序/运行支持/启动界面.sh。请保留完整项目目录结构。");
    }
    char **arguments = calloc((size_t)argc + 2, sizeof(*arguments));
    if (arguments == NULL) return show_error("内存不足，无法启动手眼标定。");
    arguments[0] = "/bin/sh";
    arguments[1] = (char *)script;
    for (int index = 1; index < argc; ++index) arguments[index + 1] = argv[index];
    pid_t application = fork();
    if (application < 0) {
        free(arguments);
        return show_error("无法创建手眼标定进程，请稍后重试。");
    }
    if (application == 0) {
        execv("/bin/sh", arguments);
        perror("无法运行启动实现");
        _exit(127);
    }
    free(arguments);
    int status;
    while (waitpid(application, &status, 0) < 0) {
        if (errno != EINTR) return show_error("无法获取手眼标定进程的退出状态。");
    }
    if (WIFEXITED(status) && WEXITSTATUS(status) == 0) return EXIT_SUCCESS;
    char message[PATH_MAX + 160];
    snprintf(message, sizeof(message), "手眼标定启动失败或异常退出，请查看具体错误记录。\n日志位置：%s/日志/运行.log", project);
    return show_error(message);
}
