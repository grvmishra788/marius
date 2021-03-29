//
// Created by Jason Mohoney on 3/28/21.
//

#include <gtest/gtest.h>
#include <marius.h>
#include <string>


/**
 * Runs marius training on a default test configuration
 */
TEST(TestMain, TestDefaultConfig) {
    std::string conf_str = std::string(MARIUS_TEST_DIRECTORY) + "/test_configs/default.ini";
    const char* conf = conf_str.c_str();
    int num_args = 2;
    const char* n_argv[] = {"marius_train", conf};
    marius(num_args, (char **)(n_argv));
}
