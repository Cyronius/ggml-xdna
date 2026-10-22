// zest: runs a llama.cpp program with the NPU add-on, with nothing to set.
//
//   zest [zest's options] llama-server [its arguments]   (or: zest server ...)
//
// zest's options (--memory-gb, --min-chunk) come before the program's name,
// so they never mix with the program's own, and reach the add-on as its
// variables.
//
// It points llama.cpp at the add-on next to itself, asks llama.cpp
// whether the NPU is usable, and if it is adds -dev XDNA0,Vulkan0 -ts 0,1
// (and -ub 2048 -b 2048 for the programs people chat with). If it isn't, it adds
// nothing, so llama.cpp runs on the GPU as usual, and says why. With no model
// given, it lists the models it finds and asks which. What the user typed is
// passed on exactly as typed, and anything they set themselves is kept.
//
// Traces: XDNA-LAUNCHER

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <shellapi.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <regex>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace {

// Text is UTF-8 throughout; only the calls into Windows below see anything else.
fs::path to_path(const std::string & s) { return fs::u8path(s); }
std::string text(const fs::path & p) { return p.u8string(); }

bool is_file(const fs::path & p) {
    std::error_code ec;
    return fs::is_regular_file(p, ec);
}

//
// the operating system
//

#ifdef _WIN32

const char * const EXE = ".exe";
const char * const ADDON = "ggml-zest.dll";

std::wstring widen(const std::string & s) {
    const int len = MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), nullptr, 0);
    std::wstring w((size_t) len, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.data(), (int) s.size(), w.data(), len);
    return w;
}

std::string narrow(const std::wstring & w) {
    const int len = WideCharToMultiByte(CP_UTF8, 0, w.data(), (int) w.size(), nullptr, 0, nullptr, nullptr);
    std::string s((size_t) len, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.data(), (int) w.size(), s.data(), len, nullptr, nullptr);
    return s;
}

// to stderr, where llama.cpp logs too
void put(const std::string & line) {
    HANDLE h = GetStdHandle(STD_ERROR_HANDLE);
    DWORD mode, n;
    if (GetConsoleMode(h, &mode)) {
        const std::wstring w = widen(line);
        WriteConsoleW(h, w.c_str(), (DWORD) w.size(), &n, nullptr);
        return;
    }
    WriteFile(h, line.data(), (DWORD) line.size(), &n, nullptr);
}

std::string env(const char * name) {
    const std::wstring wn = widen(name);
    const DWORD n = GetEnvironmentVariableW(wn.c_str(), nullptr, 0);
    if (!n) return {};
    std::wstring v(n, L'\0');
    v.resize(GetEnvironmentVariableW(wn.c_str(), v.data(), n));
    return narrow(v);
}

void set_env(const char * name, const std::string & value) {
    SetEnvironmentVariableW(widen(name).c_str(), widen(value).c_str());
}

fs::path exe_dir() {
    wchar_t buf[MAX_PATH * 4];
    const DWORD n = GetModuleFileNameW(nullptr, buf, (DWORD) std::size(buf));
    return fs::path(std::wstring(buf, n)).parent_path();
}

fs::path home() { return to_path(env("USERPROFILE")); }
fs::path llama_cache() {
    const std::string c = env("LLAMA_CACHE");
    return c.empty() ? to_path(env("LOCALAPPDATA")) / "llama.cpp" : to_path(c);
}
fs::path hf_cache() {
    const std::string h = env("HF_HOME");
    return h.empty() ? home() / ".cache" / "huggingface" / "hub" : to_path(h) / "hub";
}
fs::path state_dir() { return to_path(env("LOCALAPPDATA")) / "zest"; }

bool asking_possible() {
    DWORD mode;
    return GetConsoleMode(GetStdHandle(STD_INPUT_HANDLE), &mode);
}

// The output of `program --list-devices`, both streams; false if it didn't start.
bool list_devices(const fs::path & program, std::string & out) {
    SECURITY_ATTRIBUTES sa = { sizeof(sa), nullptr, TRUE };
    HANDLE rd = nullptr, wr = nullptr;
    if (!CreatePipe(&rd, &wr, &sa, 0)) return false;
    SetHandleInformation(rd, HANDLE_FLAG_INHERIT, 0);
    HANDLE nul = CreateFileW(L"NUL", GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, &sa, OPEN_EXISTING, 0, nullptr);

    STARTUPINFOW si = { sizeof(si) };
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = nul;
    si.hStdOutput = wr;
    si.hStdError = wr;
    PROCESS_INFORMATION pi = {};
    std::wstring cmd = L"\"" + program.wstring() + L"\" --list-devices";
    const BOOL ok = CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE, 0, nullptr, nullptr, &si, &pi);
    CloseHandle(wr);
    if (nul != INVALID_HANDLE_VALUE) CloseHandle(nul);
    if (!ok) {
        CloseHandle(rd);
        return false;
    }
    char buf[4096];
    DWORD n;
    while (ReadFile(rd, buf, sizeof(buf), &n, nullptr) && n) out.append(buf, n);
    CloseHandle(rd);
    WaitForSingleObject(pi.hProcess, 30000);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    return true;
}

// The command line after its first `skip` arguments, exactly as typed,
// split the way the C runtime splits it: argv[0] runs to its closing quote
// or the first blank; after that, quotes toggle, and backslashes escape a
// quote only in odd numbers.
const wchar_t * after_args(const wchar_t * p, int skip) {
    for (int i = 0; i < skip; i++) {
        while (*p == L' ' || *p == L'\t') p++;
        if (i == 0) {
            if (*p == L'"') {
                p++;
                while (*p && *p != L'"') p++;
                if (*p) p++;
            } else {
                while (*p && *p != L' ' && *p != L'\t') p++;
            }
            continue;
        }
        bool quoted = false;
        while (*p && (quoted || (*p != L' ' && *p != L'\t'))) {
            if (*p == L'\\') {
                const wchar_t * q = p;
                while (*q == L'\\') q++;
                if (*q == L'"') {
                    if ((q - p) % 2 == 0) quoted = !quoted;  // even: the quote is a quote; odd: a literal "
                    p = q + 1;
                } else {
                    p = q;
                }
                continue;
            }
            if (*p == L'"') {
                if (quoted && p[1] == L'"') {  // "" inside quotes: a literal quote
                    p += 2;
                    continue;
                }
                quoted = !quoted;
            }
            p++;
        }
    }
    while (*p == L' ' || *p == L'\t') p++;
    return p;
}

// One argument, quoted so the C runtime splits it back out unchanged.
std::string quote(const std::string & a) {
    if (!a.empty() && a.find_first_of(" \t\"") == std::string::npos) return a;
    std::string q = "\"";
    size_t slashes = 0;
    for (char c : a) {
        if (c == '\\') {
            slashes++;
            continue;
        }
        q.append(c == '"' ? slashes * 2 + 1 : slashes, '\\');
        slashes = 0;
        q += c;
    }
    q.append(slashes * 2, '\\');
    return q + "\"";
}

BOOL WINAPI ignore_ctrl_c(DWORD type) {
    // the program gets Ctrl+C too, and stops itself; we wait for it
    return type == CTRL_C_EVENT || type == CTRL_BREAK_EVENT;
}

// Runs the program with `added` in front of zest's own arguments from `rest` on, as typed; returns its exit code.
int run(const fs::path & exe, const std::vector<std::string> & added, int rest) {
    std::wstring cmd = L"\"" + exe.wstring() + L"\"";
    for (const std::string & a : added) cmd += L" " + widen(quote(a));
    const wchar_t * tail = after_args(GetCommandLineW(), rest);
    if (*tail) cmd += std::wstring(L" ") + tail;
    STARTUPINFOW si = { sizeof(si) };
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
    si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
    PROCESS_INFORMATION pi = {};
    if (!CreateProcessW(nullptr, cmd.data(), nullptr, nullptr, TRUE, CREATE_SUSPENDED, nullptr, nullptr, &si, &pi)) {
        put("zest: couldn't start " + text(exe) + "\n");
        return 1;
    }
    // if zest is closed or killed, the program (and anything it started) goes too
    HANDLE job = CreateJobObjectW(nullptr, nullptr);
    if (job) {
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION li = {};
        li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(job, JobObjectExtendedLimitInformation, &li, sizeof(li));
        AssignProcessToJobObject(job, pi.hProcess);
    }
    SetConsoleCtrlHandler(ignore_ctrl_c, TRUE);
    ResumeThread(pi.hThread);
    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD code = 1;
    GetExitCodeProcess(pi.hProcess, &code);
    CloseHandle(pi.hThread);
    CloseHandle(pi.hProcess);
    if (job) CloseHandle(job);
    return (int) code;
}

#else

const char * const EXE = "";
const char * const ADDON = "libggml-zest.so";

char ** g_argv = nullptr;

void put(const std::string & line) { fwrite(line.data(), 1, line.size(), stderr); }

std::string env(const char * name) {
    const char * v = getenv(name);
    return v ? v : "";
}

void set_env(const char * name, const std::string & value) { setenv(name, value.c_str(), 1); }

fs::path exe_dir() {
    std::error_code ec;
    const fs::path self = fs::read_symlink("/proc/self/exe", ec);
    return ec ? fs::absolute(g_argv[0]).parent_path() : self.parent_path();
}

fs::path home() { return to_path(env("HOME")); }
fs::path xdg(const char * name, const char * fallback) {
    const std::string v = env(name);
    return v.empty() ? home() / fallback : to_path(v);
}
fs::path llama_cache() {
    const std::string c = env("LLAMA_CACHE");
    return c.empty() ? xdg("XDG_CACHE_HOME", ".cache") / "llama.cpp" : to_path(c);
}
fs::path hf_cache() {
    const std::string h = env("HF_HOME");
    return h.empty() ? xdg("XDG_CACHE_HOME", ".cache") / "huggingface" / "hub" : to_path(h) / "hub";
}
fs::path state_dir() { return xdg("XDG_STATE_HOME", ".local/state") / "zest"; }

bool asking_possible() { return isatty(STDIN_FILENO); }

// The output of `program --list-devices`, both streams; false if it didn't start.
bool list_devices(const fs::path & program, std::string & out) {
    int fds[2];
    if (pipe(fds) != 0) return false;
    const pid_t pid = fork();
    if (pid < 0) {
        close(fds[0]);
        close(fds[1]);
        return false;
    }
    if (pid == 0) {
        const int nul = open("/dev/null", O_RDONLY);
        dup2(nul, STDIN_FILENO);
        dup2(fds[1], STDOUT_FILENO);
        dup2(fds[1], STDERR_FILENO);
        close(fds[0]);
        const std::string p = text(program);
        execl(p.c_str(), p.c_str(), "--list-devices", (char *) nullptr);
        _exit(127);
    }
    close(fds[1]);
    char buf[4096];
    ssize_t n;
    while ((n = read(fds[0], buf, sizeof(buf))) > 0 || (n < 0 && errno == EINTR))
        if (n > 0) out.append(buf, (size_t) n);
    close(fds[0]);
    int status = 0;
    waitpid(pid, &status, 0);
    return !(WIFEXITED(status) && WEXITSTATUS(status) == 127 && out.empty());
}

// Becomes the program, with `added` in front of zest's own arguments from `rest` on; returns only if that fails.
int run(const fs::path & exe, const std::vector<std::string> & added, int rest) {
    const std::string program = text(exe);
    std::vector<char *> args = { (char *) program.c_str() };
    for (const std::string & a : added) args.push_back((char *) a.c_str());
    for (char ** p = g_argv + rest; *p; p++) args.push_back(*p);
    args.push_back(nullptr);
    // replacing zest leaves Ctrl+C, the exit code and being killed to the program itself
    execv(program.c_str(), args.data());
    put("zest: couldn't start " + program + " (" + strerror(errno) + ")\n");
    return 127;
}

#endif

void say(const std::string & s) { put(s + "\n"); }

std::string lower(std::string s) {
    for (char & c : s) c = (char) std::tolower((unsigned char) c);
    return s;
}

//
// what the user already set
//

// llama.cpp reads "--batch_size" as "--batch-size"
std::string flag_name(std::string a) {
    if (a.rfind("--", 0) == 0) std::replace(a.begin(), a.end(), '_', '-');
    return a;
}

bool given(const std::vector<std::string> & args, std::initializer_list<const char *> flags,
           std::initializer_list<const char *> envs) {
    for (const std::string & a : args)
        for (const char * f : flags)
            if (flag_name(a) == f) return true;
    for (const char * e : envs)
        if (!env(e).empty()) return true;
    return false;
}

//
// is the NPU usable?
//

struct npu_check {
    bool usable = false;
    std::string why;  // when not
};

// Asks llama.cpp for its devices, with the add-on loaded, and reads the
// add-on's own line when it isn't offering the NPU.
npu_check check_npu(const fs::path & dir) {
    fs::path lister = dir / to_path(std::string("llama-server") + EXE);
    if (!is_file(lister)) lister = dir / to_path(std::string("llama-cli") + EXE);
    if (!is_file(lister)) return { false, std::string("no llama-server") + EXE + " next to zest" + EXE + " to ask" };
    std::string out;
    if (!list_devices(lister, out)) return { false, "couldn't start " + text(lister.filename()) };

    if (out.find("\n  XDNA0:") != std::string::npos) return { true, {} };
    const std::string tag = "xdna: not offering XDNA0: ";
    const size_t at = out.find(tag);
    if (at != std::string::npos) {
        const size_t end = out.find_first_of("\r\n", at);
        return { false, out.substr(at + tag.size(), end == std::string::npos ? std::string::npos : end - at - tag.size()) };
    }
    return { false, "the add-on didn't load (GGML_BACKEND_PATH=" + env("GGML_BACKEND_PATH") + ")" };
}

//
// the model menu
//

struct model {
    fs::path path;
    std::string name, where;
    uintmax_t size = 0;
};

// Every .gguf that loads as a model, in the usual places. Vision add-ons
// (mmproj-*) don't load on their own, and a model split over several files
// is listed by its first.
std::vector<model> find_models(const fs::path & dir) {
    const std::vector<std::pair<fs::path, std::string>> roots = {
        { dir / "models", "models" },
        { home() / ".lmstudio" / "models", "LM Studio" },
        { home() / ".cache" / "lm-studio" / "models", "LM Studio" },
        { llama_cache(), "llama.cpp" },
        { hf_cache(), "Hugging Face" },
    };
    static const std::regex split("-(\\d{5})-of-\\d{5}$", std::regex::icase);
    std::vector<model> found;
    for (const auto & [root, where] : roots) {
        std::error_code ec;
        if (!fs::is_directory(root, ec)) continue;
        std::vector<model> here;
        for (auto it = fs::recursive_directory_iterator(root, fs::directory_options::skip_permission_denied, ec);
             !ec && it != fs::recursive_directory_iterator(); it.increment(ec)) {
            if (it.depth() > 8) {
                it.disable_recursion_pending();
                continue;
            }
            const fs::path & p = it->path();
            if (lower(text(p.extension())) != ".gguf" || !is_file(p)) continue;
            const std::string stem = text(p.stem());
            if (lower(stem).rfind("mmproj", 0) == 0) continue;
            std::smatch m;
            if (std::regex_search(stem, m, split) && m[1].str() != "00001") continue;
            // who published it, when the folder says: LM Studio keeps
            // <publisher>/<model>/, the Hugging Face cache models--<publisher>--<model>/
            std::string label = where;
            const fs::path rel = p.lexically_relative(root);
            if (rel.begin() != rel.end() && std::next(rel.begin()) != rel.end()) {
                std::string top = text(*rel.begin());
                if (top.rfind("models--", 0) == 0) top = top.substr(8, top.find("--", 8) - 8);
                label += ": " + top;
            }
            std::error_code se;
            here.push_back({ p, stem, label, fs::file_size(p, se) });
        }
        std::sort(here.begin(), here.end(), [](const model & a, const model & b) { return lower(a.name) < lower(b.name); });
        found.insert(found.end(), here.begin(), here.end());
    }
    return found;
}

std::string size_text(uintmax_t b) {
    char s[32];
    if (b < 1000000000ull) snprintf(s, sizeof(s), "%llu MB", (unsigned long long) (b / 1000000));
    else snprintf(s, sizeof(s), "%.1f GB", (double) b / 1e9);
    return s;
}

// characters, not bytes, so names outside ASCII line up
size_t cols(const std::string & s) {
    size_t n = 0;
    for (unsigned char c : s) n += (c & 0xC0) != 0x80;
    return n;
}

std::string padded(std::string s, size_t w, bool left = true) {
    if (cols(s) > w) {
        size_t keep = 0;
        for (size_t n = 0; n + 1 < w && keep < s.size(); n++)
            for (keep++; keep < s.size() && ((unsigned char) s[keep] & 0xC0) == 0x80;) keep++;
        s = s.substr(0, keep) + "~";
    }
    const std::string pad(w - cols(s), ' ');
    return left ? s + pad : pad + s;
}

// A router preset listing every model found, for "all of them".
fs::path write_preset(const std::vector<model> & models) {
    const fs::path p = state_dir() / "models.ini";
    std::ofstream f(p, std::ios::binary);
    std::vector<std::string> used;
    for (const model & m : models) {
        std::string name = m.name;
        for (int i = 2; std::find(used.begin(), used.end(), lower(name)) != used.end(); i++)
            name = m.name + "-" + std::to_string(i);
        used.push_back(lower(name));
        f << "[" << name << "]\nmodel = " << text(m.path) << "\n\n";
    }
    return f ? p : fs::path();
}

enum class pick_kind { model, all, quit, none };
struct pick {
    pick_kind kind = pick_kind::none;
    fs::path path;
};

pick choose_model(const fs::path & dir, bool offer_all) {
    const std::vector<model> models = find_models(dir);
    if (models.empty()) {
        say(std::string("zest: no models found. Looked in: a models folder next to zest") + EXE +
            ", LM Studio's models folder, llama.cpp's download folder, and the Hugging Face download folder. Give "
            "one with -m <file.gguf>.");
        return { pick_kind::none, {} };
    }
    std::error_code ec;
    fs::create_directories(state_dir(), ec);
    fs::path last;
    {
        fs::path last_file = state_dir() / "last-model.txt";
        // Read the old preference until a choice is saved under Zest's name.
        if (!is_file(last_file)) last_file = state_dir().parent_path() / "ggml-xdna" / "last-model.txt";
        std::ifstream f(last_file, std::ios::binary);
        std::string line;
        if (std::getline(f, line)) last = to_path(line);
        if (!last.empty() && !is_file(last)) last.clear();
    }

    size_t w = 0;
    for (const model & m : models) w = std::max(w, cols(m.name));
    w = std::min<size_t>(w, 50);
    say("Models found:");
    if (offer_all) say("   0  all of them: llama-server's router, which loads each model when it's asked for");
    for (size_t i = 0; i < models.size(); i++)
        say(padded(std::to_string(i + 1), 4, false) + "  " + padded(models[i].name, w) + "  " +
            padded(size_text(models[i].size), 8, false) + "  " + models[i].where);

    for (;;) {
        std::string prompt = "Pick a number (";
        if (!last.empty()) prompt += "Enter: " + text(last.stem()) + "; ";
        prompt += "q: quit): ";
        put(prompt);
        char line[64];
        if (!fgets(line, sizeof(line), stdin)) return { pick_kind::quit, {} };
        std::string s(line);
        s.erase(std::remove_if(s.begin(), s.end(), [](char c) { return c == '\r' || c == '\n' || c == ' '; }), s.end());
        if (s == "q" || s == "Q") return { pick_kind::quit, {} };
        fs::path chosen;
        if (s.empty() && !last.empty()) {
            chosen = last;
        } else if (!s.empty() && std::all_of(s.begin(), s.end(), [](char c) { return c >= '0' && c <= '9'; }) && s.size() < 6) {
            const size_t k = (size_t) std::stoul(s);
            if (k == 0 && offer_all) {
                const fs::path ini = write_preset(models);
                if (!ini.empty()) return { pick_kind::all, ini };
                say("zest: couldn't write " + text(state_dir() / "models.ini"));
                continue;
            }
            if (k >= 1 && k <= models.size()) chosen = models[k - 1].path;
        }
        if (chosen.empty()) continue;
        std::ofstream f(state_dir() / "last-model.txt", std::ios::binary);
        f << text(chosen) << "\n";
        return { pick_kind::model, chosen };
    }
}

//
// zest's own options
//

void usage() {
    say("zest runs a llama.cpp program with prompts read on the NPU.\n"
        "\n"
        "  zest [zest's options] llama-server [its arguments]      (or: zest server ...)\n"
        "  zest llama-cli -m model.gguf\n"
        "\n"
        "It points llama.cpp at the NPU add-on next to it and, when the NPU can run, adds\n"
        "-dev XDNA0,Vulkan0 -ts 0,1 (plus -ub 2048 -b 2048 for llama-server, llama-cli\n"
        "and llama-completion). With no model given, it lists the models it finds.\n"
        "Anything you set yourself is kept.\n"
        "\n"
        "zest's options, before the program's name:\n"
        "  --memory-gb N   the most memory the NPU's copies of the weights may take, in GB\n"
        "                  (0.5 and the like work; 0 keeps the NPU out). Default: the\n"
        "                  memory free once the model is loaded, less 4 GB or a tenth of\n"
        "                  the machine's memory, whichever is larger, and at most 20 GB:\n"
        "                  past about 26 GB the NPU's copies can go bad. Layers that\n"
        "                  don't fit stay on the GPU.\n"
        "  --min-chunk N   the smallest piece of a prompt, in tokens, the NPU takes.\n"
        "                  Default 1024; smaller pieces go to the GPU, which read them\n"
        "                  faster in our tests. 512 also uses the NPU with llama.cpp's\n"
        "                  default chunk size.");
}

struct options {
    std::string memory_gb;  // as typed; empty when not given
    std::string min_chunk;
};

// A size in GB: a number, 0 or more.
bool is_gb(const std::string & s) {
    if (s.empty()) return false;
    char * end = nullptr;
    const double v = strtod(s.c_str(), &end);
    return *end == 0 && v >= 0 && v < 1e6;
}

// A token count: a whole number, 1 or more.
bool is_tokens(const std::string & s) {
    if (s.empty() || s.size() > 9) return false;
    return std::all_of(s.begin(), s.end(), [](char c) { return c >= '0' && c <= '9'; }) && std::stol(s) >= 1;
}

// Reads zest's options from argv[1] on, as "--name value" or "--name=value".
// Returns where the program's name is, 0 when help was asked for, or -1
// after saying what's wrong.
int read_options(const std::vector<std::string> & argv, options & o) {
    size_t i = 1;
    for (; i < argv.size(); i++) {
        const std::string & a = argv[i];
        if (lower(a) == "-h" || lower(a) == "--help") return 0;
        if (a.rfind("-", 0) != 0) break;
        const size_t eq = a.find('=');
        const std::string name = lower(a.substr(0, eq));
        if (name != "--memory-gb" && name != "--min-chunk") {
            say("zest: unknown option " + a + " (zest's options go before the program's name; zest -h lists them)");
            return -1;
        }
        std::string value;
        if (eq != std::string::npos) value = a.substr(eq + 1);
        else if (i + 1 < argv.size()) value = argv[++i];
        const std::string not_this = value.empty() ? std::string() : ", not \"" + value + "\"";
        if (name == "--memory-gb") {
            if (!is_gb(value)) {
                say("zest: --memory-gb takes a size in GB, such as 20 or 0.5 (0 keeps the NPU out)" + not_this);
                return -1;
            }
            o.memory_gb = value;
        } else {
            if (!is_tokens(value)) {
                say("zest: --min-chunk takes a number of tokens, such as 512 or 2048" + not_this);
                return -1;
            }
            o.min_chunk = value;
        }
    }
    return (int) i;
}

int launch(const std::vector<std::string> & argv) {
    options opt;
    const int at = read_options(argv, opt);  // argv[at]: the program
    if (at < 0) return 1;
    if (at == 0 || at >= (int) argv.size()) {
        usage();
        return at == 0 ? 0 : 1;
    }
    const fs::path dir = exe_dir();

    // the program: "llama-server", "llama-server.exe" or "server"
    std::string tool = argv[at];
    if (lower(tool).size() > 4 && lower(tool).substr(lower(tool).size() - 4) == ".exe") tool.resize(tool.size() - 4);
    fs::path exe = dir / to_path(tool + EXE);
    if (!is_file(exe) && is_file(dir / to_path("llama-" + tool + EXE))) {
        tool = "llama-" + tool;
        exe = dir / to_path(tool + EXE);
    }
    if (!is_file(exe)) {
        say("zest: there's no " + tool + EXE + " next to zest" + EXE + " (" + text(dir) + ")");
        return 1;
    }
    tool = lower(tool);
    const std::vector<std::string> args(argv.begin() + at + 1, argv.end());
    std::vector<std::string> added;

    // the add-on, unless the user points llama.cpp elsewhere
    if (env("GGML_BACKEND_PATH").empty()) {
        const fs::path lib = dir / ADDON;
        if (is_file(lib)) set_env("GGML_BACKEND_PATH", text(lib));
    }
    // zest's options reach the add-on as its variables, which the program
    // and anything it starts (the router's per-model servers) inherit
    std::string set;
    if (!opt.memory_gb.empty()) {
        set_env("GGML_XDNA_MAX_COPY_GB", opt.memory_gb);
        set += "; NPU weight copies limited to " + opt.memory_gb + " GB";
    }
    if (!opt.min_chunk.empty()) {
        set_env("GGML_XDNA_MIN_BATCH", opt.min_chunk);
        set += "; NPU takes chunks of " + opt.min_chunk + " tokens or more";
    }

    // which programs take which of our additions
    static const std::vector<std::string> common = { "llama-server", "llama-cli", "llama-completion",
                                                     "llama-perplexity", "llama-batched-bench", "llama-mtmd-cli",
                                                     "llama-imatrix", "llama-embedding" };
    const bool bench = tool == "llama-bench";
    const bool takes_dev = bench || std::find(common.begin(), common.end(), tool) != common.end();
    const bool chat = tool == "llama-server" || tool == "llama-cli" || tool == "llama-completion";
    const bool info = given(args, { "-h", "--help", "--usage", "--version", "--list-devices", "--completion-bash",
                                    "--license" }, {});

    if (takes_dev && !info && !given(args, { "-dev", "--device" }, { "LLAMA_ARG_DEVICE" })) {
        const npu_check c = check_npu(dir);
        if (c.usable) {
            added = { "-dev", bench ? "XDNA0/Vulkan0" : "XDNA0,Vulkan0" };
            // No layers on XDNA0, as llama.cpp would pick anyway (it reports no
            // free memory). Said outright, it also stops llama.cpp's memory
            // fitting before its layer search, which divides by zero on
            // mixture-of-experts models with XDNA0 listed: XDNA0 shares the
            // GPU's memory, so its use never changes as layers move onto it.
            // llama-bench doesn't fit unless asked.
            if (!bench && !given(args, { "-ts", "--tensor-split" }, { "LLAMA_ARG_TENSOR_SPLIT" }))
                added.insert(added.end(), { "-ts", "0,1" });
            if (chat && !given(args, { "-b", "--batch-size", "-ub", "--ubatch-size" },
                               { "LLAMA_ARG_BATCH", "LLAMA_ARG_UBATCH" }))
                added.insert(added.end(), { "-ub", "2048", "-b", "2048" });
            std::string shown;
            for (const std::string & a : added) shown += " " + a;
            say("zest: prompts on the NPU (added" + shown + set + ")");
        } else {
            say("zest: running on the GPU only: " + c.why);
        }
    }

    // a model, if none was given and someone is there to pick one
    const bool loads_model = takes_dev && tool != "llama-mtmd-cli";
    if (loads_model && !info && asking_possible() &&
        !given(args, { "-m", "--model", "-mu", "--model-url", "-hf", "-hfr", "--hf-repo", "-dr", "--docker-repo",
                       "--models-dir", "--models-preset" },
               { "LLAMA_ARG_MODEL", "LLAMA_ARG_MODEL_URL", "LLAMA_ARG_HF_REPO", "LLAMA_ARG_DOCKER_REPO",
                 "LLAMA_ARG_MODELS_DIR", "LLAMA_ARG_MODELS_PRESET" })) {
        const pick p = choose_model(dir, tool == "llama-server");
        if (p.kind == pick_kind::quit) return 0;
        if (p.kind == pick_kind::none) return 1;
        added.insert(added.end(), { p.kind == pick_kind::all ? "--models-preset" : "-m", text(p.path) });
    }

    return run(exe, added, at + 1);
}

} // namespace

#ifdef _WIN32
int wmain() {
    int argc = 0;
    LPWSTR * argv = CommandLineToArgvW(GetCommandLineW(), &argc);
    if (!argv) return 1;
    std::vector<std::string> args;
    for (int i = 0; i < argc; i++) args.push_back(narrow(argv[i]));
    LocalFree(argv);
    return launch(args);
}
#else
int main(int argc, char ** argv) {
    g_argv = argv;
    return launch(std::vector<std::string>(argv, argv + argc));
}
#endif
