//
// Created by orlando on 9/25/24.
//

#ifndef PARSE_ARGS_HPP
#define PARSE_ARGS_HPP

#include <string>
#include <map>
#include <iostream>
#include <unistd.h>

std::map<std::string, std::string> parse_args_v1(int argc, char *argv[]) {
    std::map<std::string, std::string> args;

    // Use getopt to parse command line arguments
    int opt;
    while ((opt = getopt(argc, argv, "v:l:m:h")) != -1) {  // -h 不需要参数，去掉 ':'
        switch (opt) {
            case 'v': // -v is for video file 
                args["video"] = optarg;
                break;
            case 'l': // -l is for label file
                args["label"] = optarg;
                break;
            case 'm': // -m is for model file
                args["model"] = optarg;
                break;
            case 'h': // -h is for help
                std::cout << "Usage: app -v video.mp4 -l coco.txt -m models.engine" << std::endl;
                exit(0);
            default:
                std::cerr << "Unknown option: -" << char(opt) << std::endl;
                break;
        }
    }

    return args;
}

std::map<std::string, std::string> parse_args_v2(int argc, char *argv[]) {
    std::map<std::string, std::string> args;

    // Use getopt to parse command line arguments
    int opt;
    while ((opt = getopt(argc, argv, "c:v:l:m:h")) != -1) {  // -h 不需要参数，去掉 ':'
        switch (opt) {
            case 'c': // -c is for config file
                args["config"] = optarg;
                break;
            case 'v': // -v is for video file 
                args["video"] = optarg;
                break;
            case 'l': // -l is for label file
                args["label"] = optarg;
                break;
            case 'm': // -m is for model file
                args["model"] = optarg;
                break;
            case 'h': // -h is for help
                std::cout << "Usage: app -c config.yaml or app -v video.mp4 -l coco.txt -m models.engine" << std::endl;
                exit(0);
            default:
                std::cerr << "Unknown option: -" << char(opt) << std::endl;
                break;
        }
    }

    // Check if mandatory arguments are provided
    if (args.find("config") == args.end() && args.find("video") == args.end()) {
        std::cerr << "Usage: app -c config.yaml or app -v video.mp4 -l coco.txt -m models.engine" << std::endl;
    }

    return args;
}


#endif //PARSE_ARGS_HPP
